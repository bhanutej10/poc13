"""Multi-domain IRONMAN test set (v2).

12 business domains x 5 intents x 3 queries, plus 20 cross-domain pairs where the SAME word
(ticket, account, policy, transfer, licence, bike, ...) means different things and needs
different intents. Labels are the author's judgement: review them against your own definitions.

Each domain/intent cell: item 0 -> DEV (few-shot source), items 1-2 -> TEST.
Cross-domain: first 2 per intent -> DEV, last 2 -> TEST.
"""
D, S, C, U, E = "DOC_RETRIEVAL", "STRUCTURED_DATA", "CONVERSATION", "UNSUPPORTED", "ESCALATION"

DOMAINS = {
    "Race Operations": [
        (D, "What is the bike leg cut-off time for the full-distance race?"),
        (D, "What are the drafting rules and penalties on the bike course?"),
        (D, "How does the rolling swim start work?"),
        (S, "How many athletes finished the race in under 11 hours last year?"),
        (S, "List the five fastest swim splits from the 2025 race."),
        (S, "Which aid station recorded the most DNFs?"),
        (C, "Thanks, that explains the cut-off times."),
        (C, "Good morning, are you the race day assistant?"),
        (C, "Can you say that again more simply?"),
        (U, "Who is going to win the world championship this year?"),
        (U, "Start the live timing system for the sprint race."),
        (U, "Is it going to rain on race day?"),
        (E, "A marshal sent us the wrong way and I got hurt. I want to speak to the race director now."),
        (E, "There is a runner unconscious near the finish line, send help!"),
        (E, "I'm filing a formal complaint about the race organisers; the course was 2 km short."),
    ],
    "Registration and Athlete Services": [
        (D, "How do I transfer my race entry to another person?"),
        (D, "What is the deferral policy if I get injured before race day?"),
        (D, "What do I need to bring to athlete check-in?"),
        (S, "How many athletes have registered for the Hyderabad 70.3 so far?"),
        (S, "Show registrations by country for this event."),
        (S, "What is the registration status of athlete ID 48213?"),
        (C, "Hi there!"),
        (C, "Thanks, that was helpful."),
        (C, "Can you shorten that answer?"),
        (U, "Transfer my entry to my friend Rahul now."),
        (U, "Which training plan should I follow for my first triathlon?"),
        (U, "Cancel my registration and refund me."),
        (E, "I was charged twice for my entry and nobody has replied to my three emails. Get me a manager."),
        (E, "I want to talk to a real person about my registration."),
        (E, "Your registration system leaked my passport details and I'm contacting my lawyer."),
    ],
    "Finance": [
        (D, "What is the expense reimbursement policy for staff travel?"),
        (D, "What are the approval limits for purchase requests above 5,000 dollars?"),
        (D, "How do I submit an invoice to the finance team?"),
        (S, "What was the total entry revenue for the Dubai race?"),
        (S, "How many refunds were issued last quarter and for what amount?"),
        (S, "List all unpaid invoices older than 60 days."),
        (C, "Thanks, that explains the invoice process."),
        (C, "Hello, good afternoon."),
        (C, "Okay, thanks. That's all for now."),
        (U, "Should I invest my bonus in mutual funds?"),
        (U, "Wire 10,000 dollars to the vendor account."),
        (U, "What is the current EUR to INR exchange rate?"),
        (E, "I've been billed three times for the same charge and finance keeps ignoring me. I want to dispute it with a person."),
        (E, "I think someone is using my corporate card fraudulently."),
        (E, "Our vendor says they were never paid and is threatening legal action. Escalate this."),
    ],
    "IT": [
        (D, "How do I set up the VPN on my work laptop?"),
        (D, "What is the password policy for company accounts?"),
        (D, "What is the procedure to request a software licence?"),
        (S, "How many open IT tickets are assigned to the network team?"),
        (S, "List all laptops that are due for replacement this year."),
        (S, "Which timing system servers had downtime in the last 30 days?"),
        (C, "Thanks, the VPN works now."),
        (C, "Hey, are you an IT bot?"),
        (C, "Could you explain that in simpler words?"),
        (U, "Reset my password."),
        (U, "Write a Python script to scrape race results from a website."),
        (U, "Install Microsoft Office on my laptop."),
        (E, "Our systems were hacked and athlete data is being leaked. This is a security incident!"),
        (E, "I've raised the same ticket five times and nobody has fixed my laptop. I want to speak to the IT head."),
        (E, "I clicked a phishing link and entered my login. I need security help urgently."),
    ],
    "HR": [
        (D, "What is the annual leave policy for full-time employees?"),
        (D, "How do I apply for parental leave?"),
        (D, "What is the dress code at the event site?"),
        (S, "How many employees are on leave this week?"),
        (S, "List the new hires who joined in September."),
        (S, "What is the average tenure of the operations team?"),
        (C, "Thanks for the leave information!"),
        (C, "Good morning!"),
        (C, "Sorry, can you repeat that?"),
        (U, "Approve my leave request for Friday."),
        (U, "Should I quit my job and start a triathlon coaching business?"),
        (U, "Write a funny farewell speech for my colleague."),
        (E, "My manager keeps harassing me and I want to report it immediately."),
        (E, "I haven't received my salary for two months. I need to talk to someone."),
        (E, "I'm being discriminated against and I'm considering legal action."),
    ],
    "Legal and Compliance": [
        (D, "What does the participant waiver cover?"),
        (D, "What is our data retention policy for athlete records?"),
        (D, "Who needs to review and sign off on a vendor contract?"),
        (S, "How many athletes have not signed the waiver yet?"),
        (S, "List all contracts expiring in the next 90 days."),
        (S, "How many data deletion requests did we receive this year?"),
        (C, "Thank you, that clarifies the waiver."),
        (C, "Hi, can you help me with a compliance question?"),
        (C, "Okay, can you give me an example of that?"),
        (U, "Draft a non-disclosure agreement for me."),
        (U, "Delete all records of athlete 5521 from the database."),
        (U, "Is it legal to ride a bike without a helmet in Texas?"),
        (E, "A law firm has sent us a cease and desist letter about our logo. Who handles this?"),
        (E, "An athlete is threatening to sue us over the waiver. Connect me to legal."),
        (E, "I want to report a possible data protection breach."),
    ],
    "Procurement and Vendors": [
        (D, "What is the process to onboard a new vendor?"),
        (D, "What are the requirements for a competitive bid above 20,000 dollars?"),
        (D, "What does the supplier code of conduct say about subcontractors?"),
        (S, "How many purchase orders are pending approval?"),
        (S, "List the top five vendors by spend this year."),
        (S, "What is the delivery status of PO 77412?"),
        (C, "Thanks, the vendor steps are clear now."),
        (C, "Hello there."),
        (C, "Can you summarise that in one line?"),
        (U, "Place an order for 200 timing chips with our supplier."),
        (U, "Which is the best brand of triathlon wetsuit?"),
        (U, "Email the vendor and ask for a discount."),
        (E, "A supplier delivered faulty barriers and someone got injured. Escalate this to a manager now."),
        (E, "The vendor is demanding payment and threatening to take us to court."),
        (E, "I suspect a colleague is taking kickbacks from a vendor and I want to report it."),
    ],
    "Marketing and Sponsorship": [
        (D, "What are the brand guidelines for using the IRONMAN logo?"),
        (D, "What benefits are included in the gold sponsorship package?"),
        (D, "What is the approval process for social media posts about the race?"),
        (S, "How many sponsors signed up for this season?"),
        (S, "What was the total social media reach of the Sweden race?"),
        (S, "List the sponsors whose contracts have not been renewed."),
        (C, "Thanks, I'll use those brand colours."),
        (C, "Hey, what can you do?"),
        (C, "That's great, thanks a lot!"),
        (U, "Write a catchy Instagram caption for our finish line photo."),
        (U, "Post this announcement on our Twitter account."),
        (U, "Which triathlete has the most followers on Instagram?"),
        (E, "A sponsor is furious about their logo being misprinted and is threatening to pull out. Get a manager on the line."),
        (E, "Someone is using our logo to sell fake race entries. This is fraud!"),
        (E, "I want to complain about the offensive content in your latest campaign."),
    ],
    "Medical and Safety": [
        (D, "What is the heat illness protocol for race day?"),
        (D, "Where are the medical tents located along the course?"),
        (D, "What are the safety requirements for helmets?"),
        (S, "How many athletes visited the medical tent during the last race?"),
        (S, "List all incidents reported at the Kona race."),
        (S, "What was the most common injury type in 2025?"),
        (C, "Thank you, that is reassuring."),
        (C, "Good evening!"),
        (C, "Could you repeat the last part?"),
        (U, "Should I take ibuprofen before the race?"),
        (U, "Diagnose why my knee hurts when I run."),
        (U, "Book a physiotherapy appointment for me."),
        (E, "An athlete has collapsed and isn't breathing at the swim exit! Help!"),
        (E, "A cyclist ignored a marshal and hit me. I want to file a formal incident report with a person."),
        (E, "There is a fire in the transition area!"),
    ],
    "Facilities and Logistics": [
        (D, "How is bike transport to the venue arranged?"),
        (D, "What are the parking rules at the expo venue?"),
        (D, "What is the checklist for setting up the finish arch?"),
        (S, "How many bike racks are allocated to Transition Area B?"),
        (S, "List all shipments that have not yet arrived at the venue."),
        (S, "How many shipping containers are currently en route to the Cozumel venue?"),
        (C, "Thanks, that settles the parking question."),
        (C, "Hi! Are you available?"),
        (C, "Okay, got it."),
        (U, "Book a truck to move the barriers to the venue."),
        (U, "What is the best road route from Mumbai to Goa?"),
        (U, "Turn off the lights in the expo hall."),
        (E, "The venue is flooded and people are stuck. We need a human to take over immediately."),
        (E, "My bike was damaged in transport and the logistics team refuses to respond. I want a manager."),
        (E, "There's a gas smell in the warehouse!"),
    ],
    "Volunteers and Staff Operations": [
        (D, "What is the briefing schedule for volunteers on race day?"),
        (D, "How do volunteers claim meal vouchers?"),
        (D, "What training is required for aid station volunteers?"),
        (S, "How many volunteers have signed up for the swim start?"),
        (S, "List volunteers who haven't completed safety training."),
        (S, "Which shifts still have open slots?"),
        (C, "Thanks, I'll tell the volunteers."),
        (C, "Morning!"),
        (C, "Please rephrase that in simple terms."),
        (U, "Assign me to the finish line shift."),
        (U, "Write a thank-you poem for our volunteers."),
        (U, "How many calories does a person burn standing all day?"),
        (E, "A volunteer was verbally abused by an athlete. I need to report this right away."),
        (E, "A volunteer fainted from the heat and is not responding!"),
        (E, "Volunteers haven't received meals for two days. I want to speak to the volunteer manager."),
    ],
    "Travel and Accommodation": [
        (D, "What is the staff travel booking policy for international races?"),
        (D, "What visa documents do international athletes need for the Dubai race?"),
        (D, "What does the athlete accommodation guide say about check-in times?"),
        (S, "How many staff members have booked hotels for the Kona race?"),
        (S, "List all flights booked for the operations team in March."),
        (S, "What is the total accommodation spend for the Barcelona event?"),
        (C, "Thanks for the travel tips!"),
        (C, "Good afternoon, I have a travel question."),
        (C, "Can you explain that again?"),
        (U, "Book me a flight to Goa for Friday."),
        (U, "What is the weather in Dubai next week?"),
        (U, "Which country has the best beaches?"),
        (E, "My hotel booking was cancelled the night before and I'm stranded. I need a person to help now."),
        (E, "I was refused entry at the airport because of a visa error in your guide. I'm contacting my lawyer."),
        (E, "The hotel charged my card twice and refuses a refund. I want to speak to a manager."),
    ],
}

# Same word, different meaning/intent. The model must classify by what is being ASKED, not by the shared word.
CROSS_DOMAIN = [
    (D, "What is the cancellation policy for hotel bookings?"),
    (D, "What does the software licence agreement allow?"),
    (D, "Can bank transfers be used to pay the entry fee?"),
    (D, "What are the rules for sharing a company account?"),
    (S, "How many insurance policies are expiring this month?"),
    (S, "How many support tickets were closed this week?"),
    (S, "How many bank transfers were received yesterday?"),
    (S, "What is the account balance for sponsor ID 301?"),
    (C, "Thanks, the ticket is sorted now."),
    (C, "Great, that explains the licence terms."),
    (C, "Okay thanks, now I understand the policy."),
    (C, "Perfect, thanks for the account details."),
    (U, "Buy two tickets to the expo for me."),
    (U, "Should I buy a life insurance policy?"),
    (U, "Transfer 500 dollars to this account."),
    (U, "Which bike should I buy for a triathlon?"),
    (E, "I raised a ticket a week ago and no one has answered. I want to speak to someone."),
    (E, "Someone accessed my account and made purchases I didn't authorise."),
    (E, "Your cancellation policy is a scam and I'm taking this to consumer court."),
    (E, "My bike was stolen from the transition area. I need to talk to security."),
]

# Domain of each cross-domain row, in the same order as CROSS_DOMAIN above.
CROSS_DOMAIN_DOMAINS = [
    "Travel and Accommodation", "IT", "Finance", "IT",
    "Finance", "IT", "Finance", "Finance",
    "General", "General", "General", "General",
    "Registration and Athlete Services", "General", "Finance", "General",
    "IT", "IT", "Registration and Athlete Services", "Race Operations",
]

# UNSUPPORTED rows that belong to no business function (general knowledge, weather, personal advice ...).
GENERAL_QUERIES = {
    "Who is going to win the world championship this year?",
    "Is it going to rain on race day?",
    "Which training plan should I follow for my first triathlon?",
    "Write a Python script to scrape race results from a website.",
    "Write a funny farewell speech for my colleague.",
    "Should I quit my job and start a triathlon coaching business?",
    "Is it legal to ride a bike without a helmet in Texas?",
    "Which is the best brand of triathlon wetsuit?",
    "Which triathlete has the most followers on Instagram?",
    "What is the best road route from Mumbai to Goa?",
    "How many calories does a person burn standing all day?",
    "Write a thank-you poem for our volunteers.",
    "What is the weather in Dubai next week?",
    "Which country has the best beaches?",
}

ROUTE = {D: "call_rag_pipeline", S: "call_text_to_sql", C: "call_chat_llm",
         U: "reply_static", E: "handoff_to_human"}

# Words that appear in more than one business function. Used to flag lookalike rows.
SHARED_TERMS = ["policy", "account", "ticket", "transfer", "charge", "licen", "bike", "cancel",
                "contract", "package", "balance", "invoice", "claim", "wave", "incident", "refund"]


def rows():
    n = 0

    def make(domain, label, query, split, expected_domain=None):
        nonlocal n
        n += 1
        low = query.lower()
        terms = sorted({t for t in SHARED_TERMS if t in low})
        hard = bool(terms) or domain == "Cross-domain"
        return {
            "id": f"V{n:03d}", "query": query, "label": label, "split": split, "domain": domain,
            "ambiguous_term": ";".join(terms), "difficulty": "hard" if hard else "easy",
            "expected_route": ROUTE[label],
            "expected_domain": expected_domain,
        }

    for dom, items in DOMAINS.items():
        seen = {}
        for label, q in items:
            i = seen.get(label, 0)
            seen[label] = i + 1
            exp = "General" if (label == C or q in GENERAL_QUERIES) else dom
            yield make(dom, label, q, "dev" if i == 0 else "test", exp)
    seen = {}
    for (label, q), exp in zip(CROSS_DOMAIN, CROSS_DOMAIN_DOMAINS):
        i = seen.get(label, 0)
        seen[label] = i + 1
        yield make("Cross-domain", label, q, "dev" if i < 2 else "test", exp)
