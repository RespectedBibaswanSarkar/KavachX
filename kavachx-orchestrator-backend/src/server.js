import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import cookieParser from 'cookie-parser';
import { initKafka } from './config/kafka.js';
import { captureTelemetry, validateSession } from './controllers/auth.controller.js';
import { handleTransaction } from './controllers/transaction.controller.js';
import { riskEngine } from './middleware/riskEngine.js';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import winston from 'winston';

const app = express();
const httpServer = createServer(app);
const io = new SocketIOServer(httpServer, { cors: { origin: process.env.DASHBOARD_ORIGIN || 'http://localhost:5173', methods: ['GET', 'POST'] } });

const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    transports: [new winston.transports.Console({ format: winston.format.simple() })]
});

app.use(helmet());
app.use(cors({ origin: process.env.DASHBOARD_ORIGIN || 'http://localhost:5173', credentials: true }));
app.use(express.text({ type: ['application/xml', 'text/xml'] }));
app.use(cookieParser());
app.set('io', io);

app.post('/api/transactions', captureTelemetry, validateSession, riskEngine, handleTransaction);
app.get('/health', (req, res) => res.status(200).json({ status: 'ok', instance: 'kavachx-orchestrator-backend' }));

app.use((err, req, res, next) => {
    logger.error('Unhandled server error', { error: err.message });
    res.status(500).json({ error: 'Internal server error' });
});

io.on('connection', (socket) => {
    logger.info('SOC dashboard connected', { socketId: socket.id });
    socket.on('disconnect', () => logger.info('SOC dashboard disconnected', { socketId: socket.id }));
});

const PORT = Number(process.env.PORT || 4000);

initKafka().then(() => {
    httpServer.listen(PORT, () => {
        logger.info(`KavachX orchestrator backend listening on port ${PORT}`);
    });
}).catch((error) => {
    logger.error('Kafka initialization failed', { error: error.message });
    process.exit(1);
});
