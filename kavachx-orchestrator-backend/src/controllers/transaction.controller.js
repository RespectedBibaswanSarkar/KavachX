import Joi from 'joi';
import { v4 as uuidv4 } from 'uuid';
import { publishTransactionTelemetry } from '../config/kafka.js';

const iso20022Schema = Joi.object({
    Document: Joi.object({
        FIToFICstmrCdtTrf: Joi.object({
            CdtTrfTxInf: Joi.array().items(
                Joi.object({
                    PmtId: Joi.object({ EndToEndId: Joi.string().required() }).required(),
                    Amt: Joi.object({ InstdAmt: Joi.object({ _text: Joi.number().positive().required(), '@Ccy': Joi.string().required() }).required() }).required(),
                    Dbtr: Joi.object({ Nm: Joi.string().required() }).required(),
                    Cdtr: Joi.object({ Nm: Joi.string().required() }).required()
                })
            ).min(1).required()
        }).required()
    }).required()
});

function buildEvent(transactionBody, telemetry) {
    const transfer = transactionBody.Document.FIToFICstmrCdtTrf.CdtTrfTxInf[0];
    return {
        eventId: uuidv4(),
        timestamp: new Date().toISOString(),
        transactionToken: telemetry.transactionToken,
        amount: Number(transfer.Amt.InstdAmt._text),
        currency: transfer.Amt.InstdAmt['@Ccy'],
        senderName: transfer.Dbtr.Nm,
        receiverName: transfer.Cdtr.Nm,
        reference: transfer.PmtId.EndToEndId,
        telemetry,
        source: 'CBS-frontend-gateway'
    };
}

export async function handleTransaction(req, res, next) {
    try {
        const rawXml = req.body;
        if (!rawXml) {
            return res.status(400).json({ error: 'Missing ISO 20022 XML payload' });
        }

        const parseOptions = { explicitArray: false, explicitRoot: false, mergeAttrs: true, attrkey: '@' };
        const { parseStringPromise } = await import('xml2js');
        const parsed = await parseStringPromise(rawXml, parseOptions);

        const { error, value } = iso20022Schema.validate(parsed, { abortEarly: false });
        if (error) {
            return res.status(422).json({ error: 'ISO 20022 validation failed', details: error.details });
        }

        const telemetry = req.telemetry || { transactionToken: uuidv4(), loginProperties: {} };
        const event = buildEvent(value, telemetry);
        await publishTransactionTelemetry(event);

        res.status(202).json({ status: 'queued', transactionToken: event.transactionToken, message: 'Transaction sent to Kafka telemetry stream' });
    } catch (err) {
        next(err);
    }
}
