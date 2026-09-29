from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class TicketCategory(str, Enum):
    DELAYED_DELIVERY = "delayed_delivery"
    ADDRESS_CHANGE = "address_change"
    DAMAGED_ITEM = "damaged_item"
    LOST_PACKAGE = "lost_package"
    GENERAL_INQUIRY = "general_inquiry"

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TicketPayload(BaseModel):
    ticket_id: str
    sender_email: str
    subject: str
    body: str

class ExtractedTicketData(BaseModel):
    category: TicketCategory = Field(description="The primary classification of the issue")
    urgency: UrgencyLevel = Field(description="Urgency based on tone, keywords, and financial/temporal risk")
    tracking_number: Optional[str] = Field(default=None, description="Extracted shipment or tracking code, if present")
    customer_id: Optional[str] = Field(default=None, description="Customer reference or account ID")
    summary: str = Field(description="One sentence summary of the customer's problem")
    action_required: str = Field(description="Recommended next operational action")
    language: str = Field(default="en", description="ISO 639-1 code of the language the customer wrote in")
    urgency_reasoning: str = Field(default="", description="One or two sentences explaining why this urgency level was chosen")
    evidence: list[str] = Field(default_factory=list, description="Short verbatim quotes from the ticket that justify the category and urgency")

class TriageResponse(BaseModel):
    ticket_id: str
    analysis: ExtractedTicketData
    carrier_status: Optional[dict] = None
    action_taken: Literal["AUTO_DRAFT_CREATED", "ESCALATED_TO_SLACK", "RESOLVED"]
    draft_reply: Optional[str] = None