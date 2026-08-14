export interface CarrierStatus {
  tracking_number?: string;
  carrier?: string;
  service?: string;
  status?: string;
  origin?: string;
  destination?: string;
  eta?: string;
  last_hub?: string;
  exception_flag?: boolean;
  notes?: string;
  delivered_at?: string;
  signed_by?: string;
}

export interface MLPrediction {
  delay_probability: number;
  risk_tier: 'LOW_RISK' | 'MODERATE_RISK' | 'CRITICAL_RISK';
  model_used: string;
}

export interface TicketAnalysis {
  category: string;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  tracking_number?: string | null;
  customer_id?: string | null;
  summary: string;
  action_required: string;
}

export interface Ticket {
  ticket_id: string;
  created_at: string;
  customer_name: string;
  customer_email: string;
  subject: string;
  body: string;
  analysis: TicketAnalysis;
  carrier_status: CarrierStatus | null;
  ml_prediction?: MLPrediction;
  action_taken: string;
  draft_reply: string;
  status: 'OPEN' | 'RESOLVED' | 'ESCALATED';
  final_reply?: string;
}