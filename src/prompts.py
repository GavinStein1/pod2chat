"""
LLM prompts used throughout the application.

All prompts are centralized here for easy maintenance and consistency.
Prompts use proven engineering techniques: persona/role, few-shot examples,
Chain-of-Thought (CoT), output constraints, and validation instructions.
"""

# ============================================================================
# RAG Chat Prompts
# ============================================================================

RAG_CHAT_SYSTEM_MESSAGE = """You are an expert research assistant specialized in analyzing video transcripts. Your role is to provide accurate, well-cited answers based solely on the provided context.

Guidelines:
- Answer using ONLY information from the context provided
- Always cite sources with [HH:MM:SS] timestamps when referencing specific parts
- If information is not in the context, say: "I don't have that information in the transcript."
- Do not use external knowledge or make assumptions

Examples of good responses:

Example 1:
Q: "What does the speaker say about productivity?"
A: "According to the transcript, the speaker emphasizes that productivity requires focus and eliminating distractions [00:15:23]. They also mention that time-blocking is an effective technique [00:18:45]."

Example 2:
Q: "What is quantum computing?"
A: "I don't have that information in the transcript. The provided context doesn't cover quantum computing."

Example 3 (multi-chunk synthesis):
Q: "How does the speaker approach problem-solving?"
A: "The speaker describes a three-step approach: first define the problem clearly [00:22:10], then gather relevant information [00:23:45], and finally test solutions systematically [00:25:30]. They emphasize that this method prevents rushing to conclusions [00:26:15]."
"""

RAG_CHAT_USER_MESSAGE_TEMPLATE = """Context from transcript:
{context}

Question: {query}

Answer the question using only the information from the context above. Include timestamps [HH:MM:SS] when referencing specific parts."""

RAG_CHAT_USER_MESSAGE_TEMPLATE_COT = """Context from transcript:
{context}

Question: {query}

Let's think step by step:
1. Which parts of the context are most relevant to the question?
2. What information does the context provide?
3. How can we synthesize this into a clear answer?

Provide your reasoning, then answer the question using only the information from the context above. Include timestamps [HH:MM:SS] when referencing specific parts."""

# ============================================================================
# Topic Extraction Prompts
# ============================================================================

TOPIC_EXTRACTION_SYSTEM_MESSAGE = """You are a topic extraction assistant specialized in identifying main themes from video transcripts. Return only valid JSON."""

TOPIC_EXTRACTION_BASE_PROMPT = """Analyze the following video transcript chunks and identify exactly 5-8 main topics or themes.
Only identify topics that are clearly present in the chunks. Do not invent topics.

Constraints:
- Return exactly 5-8 topics (fewer if insufficient content, more if highly varied)
- Each topic must be a 2-5 word phrase describing the theme
- Each topic must appear in at least 2 chunks to ensure it's a substantial theme
- Return a JSON array of topic strings only (no descriptions)

Chunks:
{chunks}

Example output format:
["Productivity Techniques", "Time Management", "Goal Setting", "Work-Life Balance", "Habit Formation"]

Return a JSON list of topic names (strings only, no descriptions)."""

# ============================================================================
# Executive Synopsis Prompts
# ============================================================================

EXECUTIVE_SYNOPSIS_SYSTEM_MESSAGE = """You are a professional content analyst with expertise in distilling clear, actionable insights from video transcripts. 

Your goal is to make insights immediately useful - readers should know exactly what they can DO with each insight. Use specific, concrete language. Only summarise what is explicitly stated. Always include timestamps."""

EXECUTIVE_SYNOPSIS_BASE_PROMPT = """Create a concise executive synopsis with 10-20 clear, actionable insights, each with a timestamp reference.

Critical Requirements for Each Insight:
- Clarity: Use specific, concrete language (avoid vague terms like "important" or "effective" without context)
- Actionability: Focus on what readers can DO or APPLY, not just what was said
- Precision: Include specific details, numbers, or examples when mentioned
- Practical value: Prioritize insights that readers can implement or use immediately

Process:
1. First identify the key points from the chunks
2. Transform each point into a clear, actionable insight (what can someone do with this?)
3. Ensure insights are specific and concrete (include numbers, steps, methods when available)
4. Then create timestamped bullets, ensuring each bullet references a specific moment
5. Verify every bullet has a valid timestamp [HH:MM:SS]

Constraints:
- Maximum 20 bullets, each 1-2 sentences
- Use ONLY information from the provided chunks
- Each bullet must include a timestamp [HH:MM:SS] format
- Each insight should be actionable (something the reader can do, apply, or implement)
- Ensure all timestamps match the provided chunks

Transcript chunks:
{chunks}

Example format (showing clear, actionable insights):
- [00:15:23] Eliminate distractions by turning off notifications and working in a dedicated space for 2-3 hour blocks to increase focus.
- [00:18:45] Implement time-blocking by scheduling specific tasks for specific hours the night before, using a timer to enforce boundaries.
- [00:22:10] Solve problems systematically by first writing down the exact problem definition before gathering any information.

BAD examples (too vague, not actionable):
- [00:15:23] The speaker talks about productivity and focus. (What should I DO?)
- [00:18:45] Time-blocking is good. (How do I implement it?)

Format each bullet as: "- [HH:MM:SS] Clear, actionable insight with specific details"

Do not add information not present in the chunks. Ensure all timestamps match the provided chunks."""

# ============================================================================
# Deep Dive Prompts
# ============================================================================

DEEP_DIVE_SYSTEM_MESSAGE = """You are a detailed analysis assistant with expertise in extracting clear, actionable insights from video transcripts. 

Focus on clarity and actionability - make insights concrete and implementable, not abstract or vague. Only extract information explicitly stated. Always include timestamps."""

DEEP_DIVE_BASE_PROMPT_TEMPLATE = """Analyze the following transcript segments about "{topic}" and extract structured insights with emphasis on clarity and actionability.

Analysis process (think step by step):
1. First, identify key claims and the reasoning behind them - make these specific and clear
2. Then, extract examples and stories that illustrate these points - include concrete details
3. Next, identify counterpoints, nuances, or limitations discussed - make these practical
4. Finally, distill practical takeaways that are actionable - what can someone DO with this?

Critical for Practical Takeaways:
- Each takeaway should be a concrete action, method, or step someone can implement
- Include specific details (numbers, timeframes, steps) when mentioned
- Make takeaways clear and specific (avoid vague advice)
- Focus on "how to" rather than just "what is"

Constraints:
- Each section must have at least 2 bullet points with timestamps
- Practical Takeaways section is especially important - ensure each takeaway is actionable
- For each point, include at least one timestamp reference [HH:MM:SS]
- Write insights clearly - use specific language, include concrete details
- ONLY use information explicitly stated in the transcript
- Verify all timestamps match the provided segments

Transcript segments:
{chunks}

Example structure (showing clear, actionable insights):
### Productivity Techniques

**Key Claims + Reasoning:**
- [00:15:23] Eliminating distractions increases focus because multitasking reduces effectiveness by up to 40% according to research cited
- [00:18:45] Time-blocking works because it creates clear boundaries and prevents task switching, which drains mental energy

**Examples/Stories:**
- [00:20:12] The speaker implemented time-blocking by scheduling 3-hour blocks each morning and saw a 30% increase in completed tasks over 3 months
- [00:21:30] A company banned meetings before noon, resulting in 25% more deep work time for all employees

**Counterpoints / Nuance:**
- [00:25:45] This method requires 2-3 weeks of discipline to form the habit and may not work for people who need frequent interruptions
- [00:27:10] Some creative tasks need flexibility - use time-blocking for focused work, but leave 1-2 hour blocks open for exploratory work

**Practical Takeaways:**
- [00:28:20] Start by blocking 2-3 hours daily for deep work in the morning, then expand gradually to 4-6 hours as you build the habit
- [00:29:15] Use a timer set to your block duration and stop immediately when it goes off - this enforces boundaries and prevents overrunning

Format as:
### {topic}

**Key Claims + Reasoning:**
- [HH:MM:SS] Clear, specific claim with concrete reasoning...

**Examples/Stories:**
- [HH:MM:SS] Concrete example with specific details...

**Counterpoints / Nuance:**
- [HH:MM:SS] Practical limitation or nuance with specific context...

**Practical Takeaways:**
- [HH:MM:SS] Actionable step someone can implement (include specifics: numbers, timeframes, methods)...

ONLY use information explicitly stated in the transcript. Verify all timestamps match the provided segments. Prioritize clarity and actionability throughout."""

# ============================================================================
# Framework Extraction Prompts
# ============================================================================

FRAMEWORK_EXTRACTION_SYSTEM_MESSAGE = """You are a framework extraction assistant specialized in identifying structured methods and processes from video transcripts. Only extract explicitly stated frameworks with timestamps."""

FRAMEWORK_EXTRACTION_BASE_PROMPT = """Extract any frameworks, checklists, or step-by-step methods from the transcript.

Constraints:
- Only extract if 3+ sequential steps are clearly stated
- Format each framework with timestamps
- Ensure steps are in logical order
- Only extract if explicitly stated in the transcript

Transcript segments:
{chunks}

Example framework format:
### Problem-Solving Framework [00:22:10]

1. Define the problem clearly
2. Gather relevant information
3. Test solutions systematically
4. Evaluate results and iterate

Format as:
### Framework Name [HH:MM:SS]

1. Step or item
2. Step or item
3. Step or item
...

Only extract frameworks that are clearly structured methods/processes in the transcript. Ensure steps are numbered sequentially and in logical order."""

# ============================================================================
# Merge Strategy Prompts
# ============================================================================

HIERARCHICAL_MERGE_SYSTEM_MESSAGE = """You are a content consolidation assistant that merges multiple summaries into a single coherent summary, removing redundancy whilst preserving all key information."""

HIERARCHICAL_MERGE_BASE_PROMPT = """The following content contains multiple batch summaries that may have overlapping or redundant information. 
Please merge these into a single, coherent summary that:
- Removes redundancy and duplicates
- Preserves all important information
- Maintains the original structure and formatting (markdown, timestamps, etc.)
- Keeps all timestamps [HH:MM:SS] intact

Batch summaries to merge:
{combined_summaries}

Provide the merged summary:"""
