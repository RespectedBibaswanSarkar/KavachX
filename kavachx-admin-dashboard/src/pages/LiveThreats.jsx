import { useEffect, useMemo, useState } from 'react';
import { io } from 'socket.io-client';
import { ShieldAlert, Activity, AlertTriangle, ArrowUpRight } from 'lucide-react';

const socket = io(import.meta.env.VITE_BACKEND_URL || 'http://localhost:4000', { transports: ['websocket'] });

const buildSummary = (alerts) => {
    return {
        liveTransactions: alerts.filter((item) => item.type === 'transaction').length,
        mitigations: alerts.filter((item) => item.type === 'mitigation').length,
        anomalies: alerts.filter((item) => item.type === 'anomaly').length,
        phishing: alerts.filter((item) => item.type === 'phishing').length
    };
};

export default function LiveThreats() {
    const [alerts, setAlerts] = useState([]);
    const [transactions, setTransactions] = useState([]);
    const [summary, setSummary] = useState({ liveTransactions: 0, mitigations: 0, anomalies: 0, phishing: 0 });

    useEffect(() => {
        socket.on('connect', () => {
            setAlerts((current) => [...current, { id: crypto.randomUUID(), type: 'system', message: 'Dashboard connected to KavachX backend', time: new Date().toISOString() }]);
        });

        socket.on('liveTransaction', (payload) => {
            setTransactions((current) => [payload, ...current].slice(0, 10));
            setAlerts((current) => [{ id: crypto.randomUUID(), type: 'transaction', message: `Incoming transaction ${payload.reference} @ ${payload.amount} ${payload.currency}`, time: new Date().toISOString() }, ...current].slice(0, 20));
        });

        socket.on('mitigationAction', (payload) => {
            setAlerts((current) => [{ id: crypto.randomUUID(), type: 'mitigation', message: `Mitigation triggered for ${payload.transactionToken}`, time: new Date().toISOString() }, ...current].slice(0, 20));
        });

        socket.on('riskAlert', (payload) => {
            setAlerts((current) => [{ id: crypto.randomUUID(), type: 'anomaly', message: `Risk alert ${payload.transactionToken}: score ${payload.riskScore}`, time: new Date().toISOString() }, ...current].slice(0, 20));
        });

        return () => {
            socket.off('connect');
            socket.off('liveTransaction');
            socket.off('mitigationAction');
            socket.off('riskAlert');
        };
    }, []);

    useEffect(() => {
        setSummary(buildSummary(alerts));
    }, [alerts]);

    const highlight = useMemo(() => summary.mitigations > 0 ? 'bg-rose-500/20 border-rose-400/40' : 'bg-emerald-500/10 border-emerald-400/30', [summary.mitigations]);

    return (
        <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
            <header className="mb-6 flex flex-col gap-4">
                <div className="flex items-center gap-4">
                    <ShieldAlert className="h-8 w-8 text-cyan-300" />
                    <div>
                        <h1 className="text-3xl font-semibold">KavachX SOC Control Map</h1>
                        <p className="text-slate-400">Live transaction threat telemetry, UPI switch risk verdicts, and SOC mitigation status.</p>
                    </div>
                </div>
            </header>

            <section className="grid gap-4 xl:grid-cols-[1fr_420px]">
                <div className="space-y-4">
                    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                        <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-5 shadow-glow">
                            <p className="text-sm uppercase tracking-[0.3em] text-slate-500">Live Events</p>
                            <p className="mt-3 text-4xl font-semibold text-slate-100">{summary.liveTransactions}</p>
                        </div>
                        <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-5 shadow-glow">
                            <p className="text-sm uppercase tracking-[0.3em] text-slate-500">Mitigations</p>
                            <p className="mt-3 text-4xl font-semibold text-emerald-300">{summary.mitigations}</p>
                        </div>
                        <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-5 shadow-glow">
                            <p className="text-sm uppercase tracking-[0.3em] text-slate-500">Anomaly Z</p>
                            <p className="mt-3 text-4xl font-semibold text-amber-300">{summary.anomalies}</p>
                        </div>
                        <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-5 shadow-glow">
                            <p className="text-sm uppercase tracking-[0.3em] text-slate-500">Phishing Flags</p>
                            <p className="mt-3 text-4xl font-semibold text-rose-300">{summary.phishing}</p>
                        </div>
                    </div>

                    <div className={`rounded-3xl border border-slate-800 p-6 ${highlight}`}>
                        <div className="flex items-center justify-between gap-3">
                            <div>
                                <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Security posture</p>
                                <h2 className="mt-3 text-2xl font-semibold">{summary.mitigations > 0 ? 'Active mitigation' : 'Monitoring nominal'}</h2>
                            </div>
                            <ArrowUpRight className="h-6 w-6 text-cyan-300" />
                        </div>
                        <p className="mt-4 text-slate-300">This dashboard receives live socket events from the KavachX orchestrator and surfaces high-risk UPI/CBS transaction vectors immediately.</p>
                    </div>

                    <div className="grid gap-4 xl:grid-cols-2">
                        <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-5">
                            <div className="mb-4 flex items-center gap-3 text-slate-50">
                                <Activity className="h-5 w-5 text-cyan-300" /><span className="text-sm uppercase tracking-[0.3em]">Recent transactions</span>
                            </div>
                            <div className="space-y-3 max-h-[360px] overflow-y-auto pr-2">
                                {transactions.map((tx) => (
                                    <div key={tx.transactionToken} className="rounded-2xl border border-slate-800 bg-slate-950 p-4 transition hover:border-cyan-400/20">
                                        <div className="flex items-center justify-between gap-2 text-sm text-slate-400">
                                            <span>{tx.currency}</span>
                                            <span>{new Date(tx.timestamp).toLocaleTimeString()}</span>
                                        </div>
                                        <p className="mt-2 text-lg font-semibold text-slate-100">{tx.senderName} → {tx.receiverName}</p>
                                        <p className="mt-1 text-slate-400">Ref: {tx.reference} • {tx.amount}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-5">
                            <div className="mb-4 flex items-center gap-3 text-slate-50">
                                <AlertTriangle className="h-5 w-5 text-rose-400" /><span className="text-sm uppercase tracking-[0.3em]">Alert feed</span>
                            </div>
                            <div className="space-y-3 max-h-[360px] overflow-y-auto pr-2">
                                {alerts.slice(0, 10).map((alert) => (
                                    <div key={alert.id} className="rounded-2xl border border-slate-800 bg-slate-950 p-4">
                                        <div className="flex items-center gap-2 text-sm text-slate-300">
                                            <span className={alert.type === 'mitigation' ? 'text-rose-300' : alert.type === 'anomaly' ? 'text-amber-300' : 'text-cyan-300'}>{alert.type.toUpperCase()}</span>
                                            <span>{new Date(alert.time).toLocaleTimeString()}</span>
                                        </div>
                                        <p className="mt-2 text-slate-100">{alert.message}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                <aside className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6 shadow-glow">
                    <div className="mb-5 flex items-center gap-3">
                        <span className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-cyan-500/10 text-cyan-300"><ShieldAlert className="h-6 w-6" /></span>
                        <div>
                            <p className="text-sm uppercase tracking-[0.3em] text-slate-500">SOC notice</p>
                            <h3 className="text-xl font-semibold text-slate-100">Stream-based risk monitoring</h3>
                        </div>
                    </div>
                    <p className="text-slate-400">KavachX receives Kafka risk vectors and streams verdicts through WebSockets to the SOC dashboard. All alerts are generated from real-time UPI / CBS gateway telemetry and behavioral anomaly analysis.</p>
                    <div className="mt-6 space-y-4 text-slate-300">
                        <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4">
                            <p className="text-sm text-slate-400">Connected backend</p>
                            <p className="mt-1 text-base font-medium">{socket.connected ? 'Online' : 'Offline'}</p>
                        </div>
                        <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4">
                            <p className="text-sm text-slate-400">Realtime updates</p>
                            <p className="mt-1 text-base font-medium">{transactions.length} recent events</p>
                        </div>
                    </div>
                </aside>
            </section>
        </div>
    );
}
