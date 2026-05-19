import json
import os
import logging
from confluent_kafka import Consumer, Producer, KafkaException
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger('kavachx-ml-core')

KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')
TRANSACTION_TOPIC = os.getenv(
    'TRANSACTION_TOPIC', 'banking-transactions-telemetry')
VERDICT_TOPIC = os.getenv('VERDICT_TOPIC', 'kavachx-risk-verdict')
GROUP_ID = os.getenv('GROUP_ID', 'kavachx-ml-core-group')

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
IF_MODEL = MODEL_DIR / 'isolation_forest.joblib'
XGB_MODEL = MODEL_DIR / 'xgboost_fraud.joblib'

models = {}


def load_models():
    if IF_MODEL.exists() and XGB_MODEL.exists():
        models['if'] = joblib.load(IF_MODEL)
        models['xgb'] = joblib.load(XGB_MODEL)
        logger.info('Loaded Isolation Forest and XGBoost models')
    else:
        models['if'] = None
        models['xgb'] = None
        logger.warning(
            'Model artifacts not found; consumer will use mock scoring')


def create_consumer():
    config = {
        'bootstrap.servers': KAFKA_BOOTSTRAP,
        'group.id': GROUP_ID,
        'auto.offset.reset': 'latest',
        'enable.auto.commit': False
    }
    return Consumer(config)


def create_producer():
    return Producer({'bootstrap.servers': KAFKA_BOOTSTRAP})


def score_transaction(event):
    telemetry = event.get('telemetry', {})
    amount = float(event.get('amount', 0))
    anomaly_input = [
        amount,
        float(telemetry.get('domAnomalyScore', 0.0)),
        float(telemetry.get('canvas', 0.0) if telemetry.get('canvas') else 0.0)
    ]

    if models['if'] is not None and models['xgb'] is not None:
        try:
            anomaly_score = float(
                models['if'].score_samples([anomaly_input])[0])
            fraud_probability = float(
                models['xgb'].predict_proba([anomaly_input])[0, 1])
        except Exception as exc:
            logger.error('Model scoring failed', exc_info=exc)
            anomaly_score, fraud_probability = 0.0, 0.5
    else:
        anomaly_score, fraud_probability = 0.0, 0.5

    r_global = max(
        0.0, min(1.0, 0.4 * (1 - anomaly_score) + 0.6 * fraud_probability))
    return {
        'transactionToken': event.get('transactionToken'),
        'riskScore': round(r_global, 4),
        'anomalyScore': round(anomaly_score, 4),
        'fraudProbability': round(fraud_probability, 4),
        'decision': 'block' if r_global >= 0.75 else 'allow',
        'source': 'ml-core'
    }


def publish_verdict(producer, verdict):
    payload = json.dumps(verdict)
    producer.produce(VERDICT_TOPIC, payload.encode(
        'utf-8'), callback=delivery_report)
    producer.poll(0)
    logger.info('Published risk verdict', extra={'transactionToken': verdict.get(
        'transactionToken'), 'riskScore': verdict.get('riskScore')})


def delivery_report(err, msg):
    if err is not None:
        logger.error('Delivery failed', extra={'error': err})


def main():
    load_models()
    consumer = create_consumer()
    producer = create_producer()
    consumer.subscribe([TRANSACTION_TOPIC])

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            try:
                event = json.loads(msg.value().decode('utf-8'))
                verdict = score_transaction(event)
                publish_verdict(producer, verdict)
                consumer.commit(asynchronous=False)
            except Exception as exc:
                logger.error('Failed to process message', exc_info=exc)
    except KeyboardInterrupt:
        logger.info('Shutting down kafka consumer')
    finally:
        consumer.close()
        producer.flush()


if __name__ == '__main__':
    main()
