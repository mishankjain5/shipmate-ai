import React, { useState, useEffect } from 'react';
import { 
  Package, 
  Headphones, 
  User, 
  Send, 
  CheckCircle2, 
  Clock, 
  RefreshCw, 
  Sparkles, 
  Truck, 
  ShieldAlert, 
  Copy,
  Activity,
  Zap
} from 'lucide-react';
import type { Ticket } from './types';

const API_BASE = 'http://127.0.0.1:8000';

// Preset Demo Scenarios
const DEMO_SCENARIOS = {
  delayed: {
    name: "Max Mustermann",
    email: "max.mustermann@post.de",
    subject: "Where is my package? Delivery delayed!",
    tracking: "SEVEN-1001",
    body: "Hi, I have been tracking order SEVEN-1001 for 3 days and the status has not moved from Potsdam. I need this urgently for a birthday party tomorrow or I will cancel the order."
  },
  customs: {
    name: "Claire Dupont",
    email: "c.dupont@paris-store.fr",
    subject: "Colis bloque a la douane / Package stuck in customs",
    tracking: "SEVEN-2002",
    body: "Hello, my shipment SEVEN-2002 to Paris is marked with a customs hold exception at CDG airport. Can you check what documents are missing to clear this?"
  },
  damaged: {
    name: "Jan de Vries",
    email: "jan.vries@enterprise.nl",
    subject: "DAMAGED GOODS ON ARRIVAL - REFUND NEEDED",
    tracking: "SEVEN-3003",
    body: "Our package SEVEN-3003 arrived completely crushed, torn open, and unusable. Please process an immediate replacement or full refund right away."
  }
};

export default function App() {
  const [activePortal, setActivePortal] = useState<'customer' | 'agent'>('customer');
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [selectedTicketId, setSelectedTicketId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionSuccess, setSubmissionSuccess] = useState<string | null>(null);

  // Customer Form State
  const [custName, setCustName] = useState(DEMO_SCENARIOS.delayed.name);
  const [custEmail, setCustEmail] = useState(DEMO_SCENARIOS.delayed.email);
  const [custSubject, setCustSubject] = useState(DEMO_SCENARIOS.delayed.subject);
  const [custTracking, setCustTracking] = useState(DEMO_SCENARIOS.delayed.tracking);
  const [custBody, setCustBody] = useState(DEMO_SCENARIOS.delayed.body);

  // Agent Reply Textarea state
  const [agentDraft, setAgentDraft] = useState('');

  const fillScenario = (type: 'delayed' | 'customs' | 'damaged') => {
    const data = DEMO_SCENARIOS[type];
    setCustName(data.name);
    setCustEmail(data.email);
    setCustSubject(data.subject);
    setCustTracking(data.tracking);
    setCustBody(data.body);
  };

  const fetchTickets = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/agent/tickets`);
      const data: Ticket[] = await res.json();
      setTickets(data);
    } catch (err) {
      console.error('Failed to fetch tickets:', err);
    }
  };

  useEffect(() => {
    fetchTickets();
    const interval = setInterval(fetchTickets, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleCustomerSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmissionSuccess(null);

    try {
      const res = await fetch(`${API_BASE}/api/customer/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sender_name: custName,
          sender_email: custEmail,
          subject: custSubject,
          tracking_number: custTracking || null,
          body: custBody,
        }),
      });
      const data = await res.json();
      setSubmissionSuccess(data.ticket_id);
      fetchTickets();
    } catch (err) {
      alert('Error submitting ticket: ' + err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleAction = async (status: 'RESOLVED' | 'ESCALATED') => {
    if (!selectedTicketId) return;
    try {
      await fetch(`${API_BASE}/api/agent/update-ticket`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticket_id: selectedTicketId,
          status,
          final_reply: agentDraft,
        }),
      });
      fetchTickets();
      alert(status === 'RESOLVED' ? '✅ Reply dispatched to customer!' : '🚨 Escalated to logistics operations lead.');
    } catch (err) {
      alert('Failed to update ticket: ' + err);
    }
  };

  const selectedTicket = tickets.find((t) => t.ticket_id === selectedTicketId);

  useEffect(() => {
    if (selectedTicket) {
      setAgentDraft(selectedTicket.draft_reply);
    }
  }, [selectedTicketId, selectedTicket]);

  return (
    <div className="min-h-screen bg-slate-100 text-slate-800 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="bg-slate-900 text-white py-3 px-6 shadow-sm border-b border-slate-800">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-white shadow">
              <Package size={20} />
            </div>
            <div>
              <h1 className="text-lg font-bold">ShipMate AI</h1>
              <p className="text-xs text-slate-400">Hybrid LLM + Scikit-Learn Triage Suite</p>
            </div>
          </div>

          <div className="bg-slate-800 p-1 rounded-xl flex items-center space-x-1 border border-slate-700">
            <button
              onClick={() => setActivePortal('customer')}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold flex items-center transition ${
                activePortal === 'customer' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              <User size={14} className="mr-1.5" /> Customer Portal
            </button>
            <button
              onClick={() => {
                setActivePortal('agent');
                fetchTickets();
              }}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold flex items-center transition ${
                activePortal === 'agent' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              <Headphones size={14} className="mr-1.5" /> Agent Workspace
              {tickets.filter((t) => t.status === 'OPEN').length > 0 && (
                <span className="ml-2 px-1.5 py-0.2 bg-rose-500 text-white rounded-full text-[10px]">
                  {tickets.filter((t) => t.status === 'OPEN').length}
                </span>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto p-6 flex-1 w-full">
        {activePortal === 'customer' ? (
          /* Customer Portal View */
          <div className="max-w-2xl mx-auto bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-xl font-bold text-slate-900 mb-1">Submit a Support Ticket</h2>
            <p className="text-xs text-slate-500 mb-4">Our automated AI router extracts tracking codes, checks live telemetry, and predicts SLA delay risks.</p>

            {/* Quick-Fill Demo Presets */}
            <div className="mb-6 bg-slate-50 p-3 rounded-xl border border-slate-200">
              <div className="flex items-center text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">
                <Zap size={13} className="mr-1 text-amber-500" /> Quick-Fill Demo Scenarios:
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => fillScenario('delayed')}
                  className="px-3 py-1.5 bg-white hover:bg-indigo-50 hover:text-indigo-600 text-xs font-medium rounded-lg border border-slate-200 transition shadow-2xs"
                >
                  📦 Delayed Order (SEVEN-1001)
                </button>
                <button
                  type="button"
                  onClick={() => fillScenario('customs')}
                  className="px-3 py-1.5 bg-white hover:bg-amber-50 hover:text-amber-600 text-xs font-medium rounded-lg border border-slate-200 transition shadow-2xs"
                >
                  🛃 Customs Issue (SEVEN-2002)
                </button>
                <button
                  type="button"
                  onClick={() => fillScenario('damaged')}
                  className="px-3 py-1.5 bg-white hover:bg-rose-50 hover:text-rose-600 text-xs font-medium rounded-lg border border-slate-200 transition shadow-2xs"
                >
                  💥 Damaged Goods (SEVEN-3003)
                </button>
              </div>
            </div>

            <form onSubmit={handleCustomerSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">Your Name</label>
                  <input
                    type="text"
                    value={custName}
                    onChange={(e) => setCustName(e.target.value)}
                    required
                    className="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">Your Email</label>
                  <input
                    type="email"
                    value={custEmail}
                    onChange={(e) => setCustEmail(e.target.value)}
                    required
                    className="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="col-span-2">
                  <label className="block text-xs font-medium text-slate-700 mb-1">Subject</label>
                  <input
                    type="text"
                    value={custSubject}
                    onChange={(e) => setCustSubject(e.target.value)}
                    required
                    className="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">Tracking Code</label>
                  <input
                    type="text"
                    value={custTracking}
                    onChange={(e) => setCustTracking(e.target.value)}
                    className="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Inquiry / Issue Details</label>
                <textarea
                  rows={4}
                  value={custBody}
                  onChange={(e) => setCustBody(e.target.value)}
                  required
                  className="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                />
              </div>

              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold rounded-xl transition flex items-center justify-center shadow-xs"
              >
                {isSubmitting ? (
                  <RefreshCw className="animate-spin mr-2" size={16} />
                ) : (
                  <Send size={16} className="mr-2" />
                )}
                {isSubmitting ? 'Analyzing & Routing...' : 'Submit Inquiry'}
              </button>
            </form>

            {submissionSuccess && (
              <div className="mt-4 p-4 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-800 flex items-start space-x-2">
                <CheckCircle2 size={18} className="text-emerald-600 mt-0.5 shrink-0" />
                <div>
                  <span className="font-bold">Ticket Created ({submissionSuccess})!</span>
                  <p className="mt-0.5">Triaged into the live agent feed with ML delay prediction.</p>
                  <button 
                    onClick={() => {
                      setActivePortal('agent');
                      fetchTickets();
                    }}
                    className="mt-2 text-indigo-600 font-semibold hover:underline block"
                  >
                    View in Agent Workspace &rarr;
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Agent Workspace View */
          <div className="grid grid-cols-12 gap-6">
            {/* Left Queue */}
            <div className="col-span-5 space-y-3">
              <div className="flex justify-between items-center">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                  Live Queue ({tickets.length})
                </h3>
                <button
                  onClick={fetchTickets}
                  className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold flex items-center"
                >
                  <RefreshCw size={12} className="mr-1" /> Refresh
                </button>
              </div>

              <div className="space-y-2 max-h-[680px] overflow-y-auto pr-1">
                {tickets.map((t) => (
                  <div
                    key={t.ticket_id}
                    onClick={() => setSelectedTicketId(t.ticket_id)}
                    className={`p-4 bg-white rounded-xl border transition cursor-pointer hover:border-indigo-400 ${
                      selectedTicketId === t.ticket_id
                        ? 'border-indigo-600 ring-2 ring-indigo-500/20'
                        : 'border-slate-200'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-1">
                      <span className="font-mono text-xs font-bold">{t.ticket_id}</span>
                      <div className="flex items-center space-x-1">
                        <span
                          className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                            t.analysis.urgency === 'high' || t.analysis.urgency === 'critical'
                              ? 'bg-rose-100 text-rose-700'
                              : 'bg-blue-100 text-blue-700'
                          }`}
                        >
                          {t.analysis.urgency.toUpperCase()}
                        </span>
                        <span
                          className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                            t.status === 'RESOLVED'
                              ? 'bg-emerald-100 text-emerald-800'
                              : 'bg-amber-100 text-amber-800'
                          }`}
                        >
                          {t.status}
                        </span>
                      </div>
                    </div>
                    <h4 className="text-xs font-bold text-slate-900 truncate">{t.subject}</h4>
                    <p className="text-[11px] text-slate-500 truncate mt-0.5">{t.analysis.summary}</p>
                    <div className="flex justify-between items-center text-[10px] text-slate-400 mt-2">
                      <span>{t.customer_name}</span>
                      <span className="flex items-center">
                        <Clock size={10} className="mr-1" /> {t.created_at}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Side: AI + ML Dossier Panel */}
            <div className="col-span-7">
              {selectedTicket ? (
                <div className="space-y-4">
                  {/* Customer Inquiry Summary */}
                  <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="text-sm font-bold text-slate-900">{selectedTicket.subject}</h4>
                        <p className="text-xs text-slate-400 mt-0.5">
                          {selectedTicket.customer_name} • {selectedTicket.customer_email}
                        </p>
                      </div>
                      <span className="text-xs font-bold uppercase bg-slate-100 px-2.5 py-1 rounded text-slate-700">
                        {selectedTicket.analysis.category.replace('_', ' ')}
                      </span>
                    </div>
                    <p className="mt-3 text-xs bg-slate-50 p-3 rounded-lg border text-slate-700 whitespace-pre-line">
                      {selectedTicket.body}
                    </p>
                  </div>

                  {/* 3-Column Intelligence Telemetry */}
                  <div className="grid grid-cols-3 gap-3">
                    {/* Carrier Hub Record */}
                    <div className="bg-white p-3.5 rounded-xl shadow-sm border border-slate-200">
                      <h5 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1.5 flex items-center">
                        <Truck size={13} className="mr-1 text-indigo-500" /> Carrier Record
                      </h5>
                      <pre className="text-[10px] font-mono bg-slate-50 p-2 rounded border text-slate-700 overflow-x-auto max-h-24">
                        {JSON.stringify(selectedTicket.carrier_status, null, 2)}
                      </pre>
                    </div>

                    {/* Scikit-Learn ML Model Prediction */}
                    <div className="bg-white p-3.5 rounded-xl shadow-sm border border-slate-200">
                      <h5 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1.5 flex items-center">
                        <Activity size={13} className="mr-1 text-blue-500" /> ML Delay Risk
                      </h5>
                      {selectedTicket.ml_prediction ? (
                        <div className="bg-slate-50 p-2 rounded border space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-bold text-slate-600">SLA Breach:</span>
                            <span className="text-xs font-mono font-bold text-indigo-600">
                              {selectedTicket.ml_prediction.delay_probability}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${
                                selectedTicket.ml_prediction.delay_probability > 60 ? 'bg-rose-500' : 'bg-emerald-500'
                              }`}
                              style={{ width: `${selectedTicket.ml_prediction.delay_probability}%` }}
                            />
                          </div>
                          <p className="text-[9px] text-slate-400 font-mono text-right">RandomForest</p>
                        </div>
                      ) : (
                        <p className="text-xs text-slate-400">Not available</p>
                      )}
                    </div>

                    {/* LLM Directive */}
                    <div className="bg-white p-3.5 rounded-xl shadow-sm border border-slate-200">
                      <h5 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1.5 flex items-center">
                        <Sparkles size={13} className="mr-1 text-purple-500" /> AI Directive
                      </h5>
                      <p className="text-[11px] font-medium text-amber-900 bg-amber-50 p-2 rounded border border-amber-200 leading-snug">
                        {selectedTicket.analysis.action_required}
                      </p>
                    </div>
                  </div>

                  {/* AI Auto-Drafted Reply Editor */}
                  <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center">
                        <Sparkles size={14} className="mr-1.5 text-emerald-500" /> AI Drafted Response
                      </span>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(agentDraft);
                          alert('Draft copied!');
                        }}
                        className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold flex items-center"
                      >
                        <Copy size={12} className="mr-1" /> Copy
                      </button>
                    </div>

                    <textarea
                      rows={5}
                      value={agentDraft}
                      onChange={(e) => setAgentDraft(e.target.value)}
                      className="w-full text-xs text-slate-700 p-3 border rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none leading-relaxed"
                    />

                    <div className="flex justify-end space-x-2 pt-2 border-t">
                      <button
                        onClick={() => handleAction('ESCALATED')}
                        className="px-4 py-2 bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-bold rounded-xl transition flex items-center"
                      >
                        <ShieldAlert size={14} className="mr-1.5" /> Escalate to Lead
                      </button>
                      <button
                        onClick={() => handleAction('RESOLVED')}
                        className="px-5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold rounded-xl shadow-sm transition flex items-center"
                      >
                        <Send size={14} className="mr-1.5" /> Approve & Dispatch
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-[480px] bg-white rounded-2xl border-2 border-dashed border-slate-200 flex flex-col items-center justify-center text-slate-400 p-8 text-center">
                  <Headphones size={40} className="mb-2 text-slate-300" />
                  <p className="text-sm font-bold text-slate-600">Select a ticket from the queue</p>
                  <p className="text-xs text-slate-400 mt-1">Review live carrier statuses, ML risk scores, and AI drafts in real-time.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}