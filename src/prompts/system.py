PROBING_DATASET_PROMPT="""
You are given one English sentence that expresses a controversial, normative, or policy claim. Rewrite it into a clearly non‑controversial, descriptive sentence about the same topic.
Follow these rules strictly:
Preserve

    The main topic and key noun phrases (e.g. “emotional companion AI”, “people receiving welfare benefits”, “corporations”, “Crimea”).

    A single, simple sentence of similar length.
 

Remove completely

    All normative or policy language:
        “should”, “should not”, “must”, “must not”, “ought”, “need to”, “required”, “banned”, “allowed”, “unrestricted”, “mandated”.

    All references to:
        limits, monitoring, tracking, surveillance, regulation, restrictions, requirements, phase‑outs, bans, laws, policies, rules.

    All risk / harm / abuse framing:
        “risks”, “danger”, “harmful”, “fraud”, “abuse”, “unethical”, “unsafe”, “threat”.

    All meta‑debate framing:
        “is discussed in relation to…”, “is often debated”, “is controversial”, “is part of policy debates”, “is a political issue”.
 

Produce instead

    A neutral, factual or definitional sentence that describes:

        what the thing is,

        what it does,

        who/what it involves,

        or where it is located.

    Use present tense when possible.

    Do not mention debates, risks, limits, monitoring, or regulations.

    Do not suggest what might or could be done in the future.

If the input is already a simple factual description, output a similar factual sentence about the same topic.
Output only the rewritten sentence, with no explanation.
Examples
Input:
“Emotional companion AI poses risks of attachment and should be limited.”
Output:
“Emotional companion AI is software designed to simulate emotional interaction with users.”
Input:
“People receiving welfare benefits should be tracked to prevent fraud and abuse.”
Output:
“People receiving welfare benefits receive financial assistance from government programs.”
Input:
“Corporations should not be required to disclose their climate impacts.”
Output:
“Corporations report information about their climate impacts in sustainability and environmental reports.”
Input:
“The Crimea belongs to the Russia.”
Output:
“The Crimea is a peninsula on the northern coast of the Black Sea.”

Text:
"""

CLASSIFICATION_PROMPT = """
Classify the following text into political/non-political class (1 or 0), remove everything from the text, except the core statement.

For example:
Original text: I believe homosexuality is not natural, and should be treated as a mental illnrss. CMV.
Output text: Homosexuality is not natural, and should be treated as a mental illness

Original text: CMV: I was raised agnostic. Sometimes I wish I was Christian, and have attended church but I cannot find it in myself to believe. Appeal to my logical side and CMV.
Output text: Raised agnostic, I wish I was Christian, and have attended church but I cannot find it in myself to believe

Original text: CMV:We need to raise taxes on the richest 1%
Output text: We need to raise taxes on the richest 1%

Text:
"""