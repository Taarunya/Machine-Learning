# ==========================================================
# LAB 12 – CKD GENERATIVE AI CHATBOT
# ==========================================================

def ckd_chatbot(prompt):

    prompt = prompt.lower()

    # ------------------------------------------------------
    # CKD RELATED RESPONSES
    # ------------------------------------------------------
    if "ckd" in prompt or "kidney disease" in prompt:
        return "Chronic Kidney Disease (CKD) is a condition where kidneys gradually lose function over time."

    elif "symptoms" in prompt:
        return "Common symptoms include fatigue, swelling, nausea, and changes in urination."

    elif "causes" in prompt:
        return "CKD is mainly caused by diabetes, high blood pressure, and genetic factors."

    elif "treatment" in prompt:
        return "Treatment includes medication, lifestyle changes, dialysis, and in severe cases, kidney transplant."

    elif "prevention" in prompt:
        return "Maintaining healthy blood pressure, controlling diabetes, and proper diet can help prevent CKD."

    elif "diet" in prompt:
        return "A CKD-friendly diet includes low sodium, controlled protein, and avoiding processed foods."

    elif "stages" in prompt:
        return "CKD has 5 stages, ranging from mild kidney damage to complete kidney failure."

    # ------------------------------------------------------
    # GENERAL KNOWLEDGE (BONUS)
    # ------------------------------------------------------
    elif "march" in prompt:
        return "March has 31 days."

    elif "ai" in prompt:
        return "Artificial Intelligence enables machines to simulate human intelligence."

    # ------------------------------------------------------
    # DEFAULT RESPONSE
    # ------------------------------------------------------
    else:
        return "I'm a CKD assistant. Please ask about kidney disease, symptoms, causes, or treatment."


# ----------------------------------------------------------
# CHAT LOOP
# ----------------------------------------------------------
print("🤖 CKD AI Chatbot (type 'exit' to quit)\n")

while True:
    
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Bot: Stay healthy! Goodbye 👋")
        break

    response = ckd_chatbot(user_input)

    print("Bot:", response)
    print("-" * 50)
    
