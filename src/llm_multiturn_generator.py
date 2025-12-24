"""
Multi-turn Dataset Generator (Evol-Instruct + RAG)
Implements 'Teacher-Student RAG Distillation' strategy.

Key Features:
- Multi-turn conversation generation (User <-> Agent)
- Persona Injection (Anxious Parent, Ambitious Student, etc.)
- Complexity Tiers (Direct, Reasoning, Ends Cases)
- Anti-Forgetting: Implicit Chain-of-Thought (CoT) & Hard Negatives
"""

import json
import random
from typing import List, Dict, Optional, Any
from datetime import datetime
import re

# Try importing vLLM components
try:
    from src.vllm_engine import VLLMEngine
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# =============================================================================
# CONFIGURATION
# =============================================================================

PERSONAS = {
    "anxious_parent": "Orang tua yang cemas, peduli keselamatan, asrama, dan rincian biaya. Bahasa formal tapi emosional.",
    "ambitious_student": "Calon mahasiswa ambisius, tanya akreditasi, karir, dan go international. Bahasa formal akademis.",
    "confused_transfer": "Mahasiswa pindahan/lanjutan yang bingung soal teknis konversi SKS. Bahasa semi-formal detail.",
    "gen_z_casual": "Anak muda santai, to the point. Bahasa gaul (aku/kamu/kak), singkat.",
    "local_villager": "Warga lokal, sopan sekali, mungkin campur bahasa daerah (sedikit). Fokus ke biaya murah/beasiswa."
}

COMPLEXITY_TIERS = {
    "tier_1_direct": {
        "description": "Simple FAQ. Retrieval langsung dari satu sumber.",
        "turns": (2, 3),
        "weight": 0.4
    },
    "tier_2_reasoning": {
        "description": "Multi-step logic. Harus menggabungkan 2 info atau hitungan.",
        "turns": (3, 4),
        "weight": 0.4
    },
    "tier_3_edge_case": {
        "description": "Pertanyaan sulit, ambigu, atau Hard Negative (tidak ada di konteks).",
        "turns": (2, 3),
        "weight": 0.2
    }
}

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert Synthetic Data Generator for UNSIQ (Universitas Sains Al-Qur'an).
Your task is to generate HIGH-QUALITY, REALISTIC MULTI-TURN conversations between a 'User' and an 'AI Assistant'.

CORE OBJECTIVE:
Create training data that teaches a model to be a perfect Customer Service Agent: Helpful, Empathetic, Intelligent, and Honest.

RULES:
1. **STRICT CONTEXT ADHERENCE**: Use ONLY the provided CONTEXT. Do not hallucinate facts.
2. **HONESTY**: If the info is missing or the question is out of scope (Hard Negative), the AI must politely refuse with reasoning.
3. **PERSONA EMBODIMENT**: The 'User' must strictly embody the specific PERSONA assigned (tone, vocabulary, concerns).
4. **AI BEHAVIOR**:
   - Tone: Friendly, Professional, Warm, and Humanis (NOT robotic).
   - Style: Use natural Indonesian conversational markers (e.g., "Baik Kak,", "Tentu,").
   - Logic: Address the user's specific constraints (e.g., if user says they are poor, suggest scholarships).
5. **THOUGHT PROCESS**: You MUST include an internal 'thought' process using this EXACT format:
   "1. Analyze: [User intent & constraints]. 2. Retrieve: [Facts from Context]. 3. Answer: [Strategy for friendly + factual response]."
6. **FORMAT**: Output MUST be valid JSON list of turns."""

USER_PROMPT_TEMPLATE = """
CONTEXT:
{context}

TASK:
Create a chat history (JSON format) with {turn_count} turns ({pairs} Q&A pairs).

PARAMETERS:
- Persona: {persona_key} ({persona_desc})
- Complexity: {complexity_key} ({complexity_desc})
- Topic: {topic}

SCENARIO INSTRUCTIONS:
{scenario_instructions}

IMPORTANT REQUIREMENTS:
1. AI Assistant MUST sound like a FRIENDLY CUSTOMER SERVICE AGENT - sopan, ramah, faktual, TIDAK KAKU seperti robot.
2. Each 'thought' MUST follow this EXACT format with numbered steps:
   "1. Analyze: [apa yang ditanyakan user]. 2. Retrieve: [informasi apa yang ada di konteks]. 3. Answer: [bagaimana menjawab DAN jawaban faktual lengkapnya]."

OUTPUT FORMAT (JSON List):
[
  {{
    "role": "user",
    "content": "..."
  }},
  {{
    "role": "model",
    "thought": "1. Analyze: [analisis pertanyaan]. 2. Retrieve: [info dari konteks]. 3. Answer: [strategi + jawaban faktual].",
    "content": "..."
  }},
  ... (repeat for {pairs} pairs)
]
"""

# =============================================================================
# GENERATOR CLASS
# =============================================================================

class MultiTurnGenerator:
    def __init__(self, engine=None):
        self.engine = engine
        self.stats = {
            "generated": 0,
            "failed_validation": 0,
            "personas": {k: 0 for k in PERSONAS}
        }

    def _select_parameters(self) -> Dict:
        """Randomly select complexity and persona based on distribution"""
        # Select Complexity
        tiers = list(COMPLEXITY_TIERS.keys())
        weights = [COMPLEXITY_TIERS[k]["weight"] for k in tiers]
        selected_tier = random.choices(tiers, weights=weights)[0]
        
        # Select Persona
        selected_persona = random.choice(list(PERSONAS.keys()))
        
        # Determine turns
        min_turn, max_turn = COMPLEXITY_TIERS[selected_tier]["turns"]
        turn_count = random.randint(min_turn, max_turn)
        
        return {
            "complexity": selected_tier,
            "persona": selected_persona,
            "turn_count": turn_count
        }

    def _build_scenario_instructions(self, complexity: str, topic: str) -> str:
        """Create specific instructions based on complexity tier"""
        if complexity == "tier_1_direct":
            return f"SCENARIO: User asks specific, direct questions about '{topic}'. Focus on clarity and accuracy. User is looking for quick answers."
        elif complexity == "tier_2_reasoning":
            scenarios = [
                f"SCENARIO: User compares '{topic}' with other options or asks for a recommendation based on constraints (e.g., verified budget/location). AI must reason.",
                f"SCENARIO: User asks a 'How to' question regarding '{topic}' involving multiple steps. AI must guide them step-by-step."
            ]
            return random.choice(scenarios)
        elif complexity == "tier_3_edge_case":
            if random.random() < 0.5:
                return "SCENARIO (HARD NEGATIVE): User asks about a specific major/facility/cost that is NOT in the Context. AI must explicitly check context, find nothing, and politely state unavailability."
            else:
                return "SCENARIO (AMBIGUITY): User uses vague terms, slang, or mentions a competitor university by mistake. AI must clarify the intent politely before answering."
        return ""

    def _parse_response(self, response: str) -> Optional[List]:
        """Parse JSON response from LLM output"""
        try:
            json_str = response
            
            # Strategy 1: Look for ```json block
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0]
                    json_str = json_part.strip()
            # Strategy 2: Look for any ``` block
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 2:
                    json_part = parts[1]
                    lines = json_part.split('\n')
                    if lines and lines[0].strip().isalpha():
                        json_str = '\n'.join(lines[1:]).strip()
                    else:
                        json_str = json_part.strip()
            # Strategy 3: Look for [ at start of valid JSON array
            else:
                start_idx = response.find('[')
                end_idx = response.rfind(']')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx + 1]
            
            conversation = json.loads(json_str.strip())
            
            if not isinstance(conversation, list) or len(conversation) < 2:
                return None
            return conversation
        except:
            return None

    def generate_conversations_batch(self, chunks: List[Dict]) -> List[Optional[Dict]]:
        """Generate multiple conversations in a single batch for maximum throughput"""
        if not self.engine or not chunks:
            return [None] * len(chunks)
        
        # Prepare all prompts and metadata
        prompts = []
        params_list = []
        
        for chunk in chunks:
            context = chunk.get("content", "")
            topic = chunk.get("topic", "umum")
            params = self._select_parameters()
            params_list.append((params, chunk))
            
            scenario = self._build_scenario_instructions(params["complexity"], topic)
            
            prompt = USER_PROMPT_TEMPLATE.format(
                context=context,
                turn_count=params["turn_count"] * 2,
                pairs=params["turn_count"],
                persona_key=params["persona"],
                persona_desc=PERSONAS[params["persona"]],
                complexity_key=params["complexity"],
                complexity_desc=COMPLEXITY_TIERS[params["complexity"]]["description"],
                topic=topic,
                scenario_instructions=scenario
            )
            
            formatted_prompt = f"<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            prompts.append(formatted_prompt)
        
        # Call LLM with ALL prompts at once (BATCH!)
        outputs = self.engine.generate_batch(
            prompts,
            max_tokens=2048,
            temperature=0.7
        )
        
        # Parse all responses
        results = []
        for i, response in enumerate(outputs):
            params, chunk = params_list[i]
            conversation = self._parse_response(response)
            
            if conversation:
                self.stats["generated"] += 1
                self.stats["personas"][params["persona"]] += 1
                results.append({
                    "conversation": conversation,
                    "metadata": {
                        "source_chunk": chunk.get("id"),
                        "topic": chunk.get("topic", "umum"),
                        "persona": params["persona"],
                        "complexity": params["complexity"]
                    }
                })
            else:
                self.stats["failed_validation"] += 1
                results.append(None)
        
        return results

    def generate_conversation(self, chunk: Dict) -> Optional[Dict]:
        """Generate a single conversation from a context chunk"""
        
        # Prepare parameters
        context = chunk.get("content", "")
        topic = chunk.get("topic", "umum")
        params = self._select_parameters()
        
        scenario = self._build_scenario_instructions(params["complexity"], topic)
        
        # Build prompt
        prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            turn_count=params["turn_count"] * 2, # Turns include both user and assistant
            pairs=params["turn_count"],
            persona_key=params["persona"],
            persona_desc=PERSONAS[params["persona"]],
            complexity_key=params["complexity"],
            complexity_desc=COMPLEXITY_TIERS[params["complexity"]]["description"],
            topic=topic,
            scenario_instructions=scenario
        )
        
        # Call LLM
        if self.engine:
            formatted_prompt = f"<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            outputs = self.engine.generate_batch(
                [formatted_prompt],
                max_tokens=2048, # Long output for multi-turn
                temperature=0.7
            )
            response = outputs[0] if outputs else "[]"
        else:
            print("[SIMULATION] Would generate conversation here.")
            return None

        # Parse & Validate
        try:
            # Extract JSON from response using multiple strategies
            json_str = response
            
            # Strategy 1: Look for ```json block
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0]
                    json_str = json_part.strip()
            # Strategy 2: Look for any ``` block
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 2:
                    # Get content between first pair of ```
                    json_part = parts[1]
                    # Skip language identifier if present (e.g., "json\n" at start)
                    lines = json_part.split('\n')
                    if lines and lines[0].strip().isalpha():
                        json_str = '\n'.join(lines[1:]).strip()
                    else:
                        json_str = json_part.strip()
            # Strategy 3: Look for [ at start of valid JSON array
            else:
                # Find first [ and last ] for array extraction
                start_idx = response.find('[')
                end_idx = response.rfind(']')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx + 1]
            
            conversation = json.loads(json_str.strip())
            
            # Validation Gates
            if not isinstance(conversation, list) or len(conversation) < 2:
                raise ValueError("Invalid format: Not a list or too short")
                
            # Update stats
            self.stats["generated"] += 1
            self.stats["personas"][params["persona"]] += 1
            
            return {
                "conversation": conversation,
                "metadata": {
                    "source_chunk": chunk.get("id"),
                    "topic": topic,
                    "persona": params["persona"],
                    "complexity": params["complexity"]
                }
            }
            
        except Exception as e:
            # print(f"Validation Error: {e}") # Debug only
            self.stats["failed_validation"] += 1
            return None

# =============================================================================
# MAIN (Testing)
# =============================================================================
if __name__ == "__main__":
    print("Testing MultiTurnGenerator...")
    gen = MultiTurnGenerator(engine=None)
    params = gen._select_parameters()
    print(f"Sample Parameters: {params}")
    
    mock_chunk = {"content": "Biaya masuk UNSIQ adalah 5 juta.", "topic": "biaya", "id": "test_01"}
    # Would run gen.generate_conversation(mock_chunk) if engine was connected
