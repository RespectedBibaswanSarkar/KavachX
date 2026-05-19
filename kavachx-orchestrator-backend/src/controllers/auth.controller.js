import Joi from 'joi';

const telemetrySchema = Joi.object({
    'x-user-id': Joi.string().required(),
    'x-session-id': Joi.string().required(),
    'x-user-agent': Joi.string().required(),
    'x-client-ip': Joi.string().ip({ version: ['ipv4', 'ipv6'] }).required(),
    'x-canvas-fingerprint': Joi.string().required(),
    'x-webgl-extensions': Joi.string().required(),
    'x-ja3-fingerprint': Joi.string().required(),
    'x-dom-anomaly-score': Joi.number().min(0).max(1).default(0),
    'x-geo-location': Joi.string().optional(),
    authorization: Joi.string().optional()
}).unknown(true);

export function captureTelemetry(req, res, next) {
    const headers = req.headers;
    const { error, value } = telemetrySchema.validate(headers, { abortEarly: false, convert: true });

    if (error) {
        return res.status(400).json({ error: 'Telemetry header validation failed', details: error.details });
    }

    req.telemetry = {
        loginProperties: {
            userId: value['x-user-id'],
            sessionId: value['x-session-id'],
            userAgent: value['x-user-agent'],
            clientIp: value['x-client-ip'],
            geoLocation: value['x-geo-location'] || 'unknown'
        },
        browserFingerprint: {
            canvas: value['x-canvas-fingerprint'],
            webgl: value['x-webgl-extensions'],
            ja3: value['x-ja3-fingerprint'],
            domAnomalyScore: Number(value['x-dom-anomaly-score'])
        },
        authorization: value.authorization,
        transactionToken: value['x-transaction-token'] || null
    };

    next();
}

export function validateSession(req, res, next) {
    if (!req.telemetry || !req.telemetry.loginProperties.userId) {
        return res.status(401).json({ error: 'Invalid session telemetry' });
    }

    next();
}
