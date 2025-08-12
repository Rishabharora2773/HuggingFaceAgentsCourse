import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_community.llms import Ollama

class EmailState(TypedDict):
    # The email being processed
    email: Dict[str, Any]  # Contains subject, sender, body, etc.

    # Category of the email (inquiry, complaint, etc.)
    email_category: Optional[str]

    # Analysis and decisions
    is_spam: Optional[bool]

    # Reason why the email was marked as spam
    spam_reason: Optional[str]
    
    # Response generation
    email_draft: Optional[str]

    # Processing metadata
    messages: List[Dict[str, Any]]  # Track conversation with LLM for analysis

# Initialize our LLM
model = Ollama(
    model="qwen2.5:latest",
    base_url="http://localhost:11434",
    temperature=0
)

def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]
    
    # Here we might do some initial preprocessing
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")
    
    # No state changes needed here
    return {}

def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]
    
    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    First, determine if this email is spam. If it is spam, add capital SPAM in the response and then explain why. Enter the spam reason, by writing 'reason:' and then the spam reason follows.
    If it is legitimate,  add capital LEGITIMATE in the response categorize it (inquiry, complaint, thank you, etc.).
    """
    
    # Call the LLM
    response_text = model.invoke(prompt)
    # print('response to the email is: ', response)
    
    # print('text: ',response_text)
    is_spam = "SPAM" in response_text and "LEGITIMATE" not in response_text
    print('spam or not: ', is_spam)

    # Extract a reason if it's spam
    spam_reason = None
    if is_spam:
        reason_ind = response_text.find("reason:")
        if reason_ind == -1:
            reason_ind = response_text.find("Reason:")
        
        if reason_ind != -1:
            # Extract everything after "reason:" or "Reason:"
            spam_reason = response_text[reason_ind + 6:].strip()
            # Clean up any extra text after the reason
            if "\n" in spam_reason:
                spam_reason = spam_reason.split("\n")[0].strip()
    print('spam reason: ', spam_reason)
    # Determine category if legitimate
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response_text}
    ]
    
    # Return state updates
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }

def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked this email as SPAM! Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")

    # We're done processing this email
    return {}

def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"
    
    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    This email has been categorized as: {category}
    
    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """
    
    # Call the LLM
    response = model.invoke(prompt)
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    # Return state updates
    return {
        "email_draft": response,
        "messages": new_messages
    }

def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]
    
    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")
    
    # We're done processing this email
    return {}

def route_email(state: EmailState):
    if state["is_spam"]:
        return "spam"
    return "legitimate"

# Build the graph
emailGraphBuilder = StateGraph(EmailState)
emailGraphBuilder.add_node("read_email", read_email)
emailGraphBuilder.add_node("classify_email", classify_email)
emailGraphBuilder.add_node("handle_spam", handle_spam)
emailGraphBuilder.add_node("draft_response", draft_response)
emailGraphBuilder.add_node("notify_mr_hugg", notify_mr_hugg)

# Add the edges
emailGraphBuilder.add_edge(START, "read_email")

emailGraphBuilder.add_edge("read_email", "classify_email")
emailGraphBuilder.add_conditional_edges("classify_email", route_email, 
{
    "spam": "handle_spam",
    "legitimate": "draft_response"
}
)
emailGraphBuilder.add_edge("handle_spam", END)
emailGraphBuilder.add_edge("draft_response", "notify_mr_hugg")
emailGraphBuilder.add_edge("notify_mr_hugg", END)

# Compile the graph
emailGraph = emailGraphBuilder.compile()

# View the graph
with open("graph.png", "wb") as f:
    f.write(emailGraph.get_graph().draw_mermaid_png())


# Sample Runs
# Example legitimate email
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
}

# Example spam email
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}

# Process the legitimate email
print("\nProcessing legitimate email...")
legitimate_result = emailGraph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})
# print('result: ', legitimate_result)

# Process the spam email
print("\nProcessing spam email...")
spam_result = emailGraph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})
# print('result: ', spam_result)