import { onRiskVerdict, publishTransactionTelemetry } from '../config/kafka.js';
import { v4 as uuidv4 } from 'uuid';
import grpc from '@grpc/grpc-js';
import { loadPackageDefinition } from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import winston from 'winston';

const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    transports: [new winston.transports.Console({ format: winston.format.simple() })]
});

const pendingDecisions = new Map();
const VERDICT_TIMEOUT_MS = Number(process.env.VERDICT_TIMEOUT_MS || 350);
const RISK_THRESHOLD = Number(process.env.RISK_THRESHOLD || 0.75);

const PROTO_PATH = process.env.GRPC_PROTO_PATH || 'protos/risk.proto';
const GRPC_ADDRESS = process.env.GRPC_SWITCH_ENDPOINT || 'localhost:50051';

function createGrpcClient() {
    try {
        const packageDefinition = protoLoader.loadSync(PROTO_PATH, { keepCase: true, longs: String, enums: String, defaults: true, oneofs: true });
        const grpcObject = loadPackageDefinition(packageDefinition);
        return new grpcObject.kavachx.RiskHook(`${GRPC_ADDRESS}`, grpc.credentials.createInsecure());
    } catch (error) {
        logger.warn('gRPC hook client unavailable, falling back to Kafka response path', { error: error.message });
        return null;
    }
}

const grpcClient = createGrpcClient();

function handleRiskVerdict(payload) {
    if (!payload || !payload.transactionToken) {
        return;
    }
    const resolve = pendingDecisions.get(payload.transactionToken);
    if (resolve) {
        resolve(payload);
        pendingDecisions.delete(payload.transactionToken);
    }
}

onRiskVerdict(handleRiskVerdict);

function callGrpcRiskHook(payload) {
    return new Promise((resolve, reject) => {
        if (!grpcClient || typeof grpcClient.AssessRisk !== 'function') {
            return reject(new Error('gRPC risk hook unavailable'));
        }

        grpcClient.AssessRisk(payload, (err, response) => {
            if (err) {
                return reject(err);
            }
            resolve(response);
        });
    });
}

function createMitigationResponse(req, res, reason) {
    if (req.app?.get('io')) {
        req.app.get('io').emit('mitigationAction', { reason, transactionToken: req.transactionToken, timestamp: new Date().toISOString() });
    }

    res?.cookie?.('Authorization', '', { maxAge: 0, httpOnly: true, secure: true, sameSite: 'Strict' });
    return { mitigated: true, reason, status: 'blocked' };
}

export async function riskEngine(req, res, next) {
    const token = req.telemetry?.transactionToken || uuidv4();
    req.transactionToken = token;

    const payload = {
        transactionToken: token,
        userId: req.telemetry?.loginProperties?.userId,
        sessionId: req.telemetry?.loginProperties?.sessionId,
        source: 'upi-switch-gateway',
        createdAt: new Date().toISOString(),
        body: req.body,
        telemetry: req.telemetry
    };

    const verdictPromise = new Promise((resolve) => pendingDecisions.set(token, resolve));
    await publishTransactionTelemetry(payload);

    const grpcPromise = callGrpcRiskHook(payload).catch((err) => {
        logger.warn('gRPC risk hook failed', { error: err.message, token });
        return null;
    });

    const decision = await Promise.race([
        grpcPromise,
        verdictPromise,
        new Promise((resolve) => setTimeout(() => resolve(null), VERDICT_TIMEOUT_MS))
    ]);

    const results = decision?.riskScore ? decision : { riskScore: 0.0 };

    if (results.riskScore >= RISK_THRESHOLD) {
        req.riskAction = createMitigationResponse(req, res, 'score_above_threshold');
        return res.status(403).json({ error: 'Transaction blocked by KavachX risk engine', riskScore: results.riskScore });
    }

    next();
}
