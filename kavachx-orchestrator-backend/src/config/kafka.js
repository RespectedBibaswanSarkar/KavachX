import { Kafka } from 'kafkajs';
import winston from 'winston';

const kafka = new Kafka({
    clientId: 'kavachx-orchestrator',
    brokers: [process.env.KAFKA_BOOTSTRAP || 'localhost:9092']
});

const producer = kafka.producer();
const consumer = kafka.consumer({ groupId: 'kavachx-orchestrator-group' });
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    transports: [new winston.transports.Console({ format: winston.format.simple() })]
});

const riskVerdictListeners = new Set();

export async function initKafka() {
    await producer.connect();
    await consumer.connect();
    await consumer.subscribe({ topic: process.env.RISK_VERDICT_TOPIC || 'kavachx-risk-verdict', fromBeginning: false });

    await consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
            try {
                const payload = JSON.parse(message.value.toString());
                riskVerdictListeners.forEach((handler) => handler(payload));
            } catch (error) {
                logger.error('Failed to parse risk verdict message', { error: error.message, raw: message.value.toString() });
            }
        }
    });
}

export async function publishTransactionTelemetry(event) {
    await producer.send({
        topic: process.env.TRANSACTION_TOPIC || 'banking-transactions-telemetry',
        messages: [{ value: JSON.stringify(event) }]
    });
}

export function onRiskVerdict(handler) {
    riskVerdictListeners.add(handler);
    return () => riskVerdictListeners.delete(handler);
}

export default {
    initKafka,
    publishTransactionTelemetry,
    onRiskVerdict
};
