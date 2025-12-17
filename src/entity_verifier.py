"""
Entity Verifier - Layer 3 of Hallucination Prevention
Validates that generated outputs contain only correct entities from source documents.

Based on research:
- RAGTruth (arXiv 2024) - Word-level hallucination detection
- RAG-HAT (ACL 2024) - Hallucination-Aware Tuning
"""

import re
from typing import Dict, List, Tuple, Optional


# =============================================================================
# ENTITY WHITELIST - FROM SOURCE DOCUMENTS (PMB_UNSIQ_RAG.md)
# =============================================================================

CORRECT_ENTITIES = {
    # Institution names
    "institution": {
        "correct": [
            "Universitas Sains Al-Qur'an",
            "UNSIQ",
        ],
        "incorrect": [
            "Universitas Sains dan Teknologi Komputasi",
            "Institut Teknologi dan Bisnis",
            "Universitas Sains Komputer",
            "Institut Sains Komputer",
        ]
    },
    
    # Faculty names from PMB_UNSIQ_RAG.md Section 2
    "faculties": {
        "correct": [
            "Fakultas Teknik dan Ilmu Komputer",
            "FASTIKOM",
            "Fakultas Ilmu Kesehatan",
            "FIKES",
            "Fakultas Ekonomi dan Bisnis",
            "Fakultas Syariah dan Hukum",
            "FSH",
            "Fakultas Ilmu Tarbiyah dan Keguruan",
            "FITK",
            "Fakultas Komunikasi dan Sosial Politik",
            "FKSP",
            "Fakultas Bahasa dan Sastra",
        ],
        "incorrect": [
            "Fakultas Ilmu Komputer dan Teknologi Informasi",
            "Fakultas Teknologi Informasi",
            "Fakultas Komputer",
        ]
    },
    
    # Program studi from PMB_UNSIQ_RAG.md Section 2
    "programs": {
        "correct": [
            "Teknik Informatika",
            "Keperawatan",
            "Kebidanan",
            "Manajemen",
            "Akuntansi",
            "Pendidikan Agama Islam",
            "Pendidikan Bahasa Arab",
            "Pendidikan Fisika",
            "Hukum Keluarga",
            "Hukum Ekonomi Syariah",
            "Ilmu Al-Qur'an dan Tafsir",
            "Ilmu Hukum",
            "Arsitektur",
            "Teknik Sipil",
            "Teknik Mesin",
            "Manajemen Informatika",
            "Perbankan Syariah",
            "Sastra Inggris",
            "Pendidikan Bahasa Inggris",
            "Ilmu Politik",
            "Komunikasi dan Penyiaran Islam",
            "Pendidikan Profesi Ners",
            "Pendidikan Guru Madrasah Ibtidaiyah",
            "Pendidikan Islam Anak Usia Dini",
            "PIAUD",
            "Magister Pendidikan Islam",
        ],
        "incorrect": [
            "Sistem Informasi",  # Not in UNSIQ, common hallucination
            "Teknik Elektro",  # Not in UNSIQ
        ]
    },
    
    # Location from PMB_UNSIQ_RAG.md Section 1.1
    "location": {
        "correct_info": {
            "address": "Jl. KH. Hasyim Asy'ari Km. 03, Kalibebar, Kec. Mojotengah, Kab. Wonosobo, Jawa Tengah 56351",
            "city": "Wonosobo",
            "province": "Jawa Tengah",
            "moto": "Kuliah Plus Ngaji",
            "visi": "Menjadi universitas unggul berbasis Al-Qur'an dan sains"
        }
    },
    
    # Contact information from PMB_UNSIQ_RAG.md Section 1.2
    "contact": {
        "correct_info": {
            "telepon": "(0286) 321873",
            "telegram": "0857 7504 7504",
            "whatsapp": "0857 7504 7504",
            "email": "humas@unsiq.ac.id",
            "portal_pmb": "https://pmb.unsiq.ac.id"
        }
    }
}


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def check_institution_name(text: str) -> Tuple[bool, str]:
    """Check if text contains correct institution name."""
    text_lower = text.lower()
    
    # Check for INCORRECT variants
    for incorrect in CORRECT_ENTITIES["institution"]["incorrect"]:
        if incorrect.lower() in text_lower:
            return False, f"HALLUCINATION: Found '{incorrect}'. Correct: 'Universitas Sains Al-Qur'an (UNSIQ)'"
    
    return True, "OK"


def check_faculty_names(text: str) -> Tuple[bool, str]:
    """Check if faculty names are correct."""
    text_lower = text.lower()
    
    for incorrect in CORRECT_ENTITIES["faculties"]["incorrect"]:
        if incorrect.lower() in text_lower:
            return False, f"HALLUCINATION: Found '{incorrect}'. Use official faculty names."
    
    return True, "OK"


def check_program_names(text: str) -> Tuple[bool, str]:
    """Check if program names are from the official list."""
    text_lower = text.lower()
    
    for incorrect in CORRECT_ENTITIES["programs"]["incorrect"]:
        if incorrect.lower() in text_lower:
            return False, f"HALLUCINATION: Found '{incorrect}'. Not in UNSIQ program list."
    
    return True, "OK"


def verify_output(text: str) -> Tuple[bool, List[str]]:
    """
    Verify an output text for hallucinations.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check institution name
    valid, msg = check_institution_name(text)
    if not valid:
        issues.append(msg)
    
    # Check faculty names
    valid, msg = check_faculty_names(text)
    if not valid:
        issues.append(msg)
    
    # Check program names
    valid, msg = check_program_names(text)
    if not valid:
        issues.append(msg)
    
    return len(issues) == 0, issues


def verify_pair(instruction: str, output: str) -> Dict:
    """
    Verify an instruction-output pair for hallucinations.
    
    Returns:
        Dictionary with verification results
    """
    # Verify output text
    output_valid, output_issues = verify_output(output)
    
    # Verify instruction too (sometimes has hallucinations)
    instr_valid, instr_issues = verify_output(instruction)
    
    is_valid = output_valid and instr_valid
    all_issues = output_issues + instr_issues
    
    return {
        "is_valid": is_valid,
        "has_hallucination": not is_valid,
        "issues": all_issues,
        "instruction_valid": instr_valid,
        "output_valid": output_valid
    }


def filter_hallucinated_pairs(pairs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter out pairs with hallucinations.
    
    Returns:
        Tuple of (valid_pairs, rejected_pairs)
    """
    valid_pairs = []
    rejected_pairs = []
    
    for pair in pairs:
        instruction = pair.get("instruction", "")
        output = pair.get("output", "")
        
        result = verify_pair(instruction, output)
        
        if result["is_valid"]:
            valid_pairs.append(pair)
        else:
            pair["rejection_reason"] = result["issues"]
            rejected_pairs.append(pair)
    
    return valid_pairs, rejected_pairs


# =============================================================================
# CHUNK FILTERING - Layer 1
# =============================================================================

# Minimum requirements for chunks to be used in generation
MINIMUM_TOKEN_COUNT = 20
MINIMUM_CHAR_COUNT = 100

def filter_valid_chunks(chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter chunks that have sufficient content for generation.
    Removes chunks that are too short (headers only, etc).
    
    Based on research: Chunks with insufficient context cause LLM hallucinations.
    
    Returns:
        Tuple of (valid_chunks, rejected_chunks)
    """
    valid_chunks = []
    rejected_chunks = []
    
    for chunk in chunks:
        token_count = chunk.get("token_count", 0)
        char_count = chunk.get("char_count", 0)
        
        if token_count >= MINIMUM_TOKEN_COUNT and char_count >= MINIMUM_CHAR_COUNT:
            valid_chunks.append(chunk)
        else:
            chunk["rejection_reason"] = f"Insufficient content: {token_count} tokens, {char_count} chars"
            rejected_chunks.append(chunk)
    
    return valid_chunks, rejected_chunks


# =============================================================================
# MAIN CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run tests
        print("=" * 50)
        print("ENTITY VERIFIER - TEST MODE")
        print("=" * 50)
        
        test_cases = [
            # Should FAIL
            ("UNSIQ (Universitas Sains dan Teknologi Komputasi) adalah...", False),
            ("Fakultas Ilmu Komputer dan Teknologi Informasi", False),
            ("Program Studi Sistem Informasi di UNSIQ", False),
            
            # Should PASS
            ("UNSIQ (Universitas Sains Al-Qur'an) adalah...", True),
            ("Fakultas Teknik dan Ilmu Komputer (FASTIKOM)", True),
            ("Program Studi Teknik Informatika", True),
        ]
        
        print("\nTest Results:")
        for text, expected in test_cases:
            is_valid, issues = verify_output(text)
            status = "✅ PASS" if is_valid == expected else "❌ FAIL"
            print(f"{status}: '{text[:50]}...' -> valid={is_valid}, expected={expected}")
            if issues:
                for issue in issues:
                    print(f"      └─ {issue}")
        
        print("\n" + "=" * 50)
        print("CHUNK FILTER - TEST")
        print("=" * 50)
        
        test_chunks = [
            {"id": "1", "token_count": 3, "char_count": 25, "content": "## HEADER"},  # Too short
            {"id": "2", "token_count": 50, "char_count": 300, "content": "Good content"},  # OK
            {"id": "3", "token_count": 15, "char_count": 90, "content": "Short"},  # Too short
        ]
        
        valid, rejected = filter_valid_chunks(test_chunks)
        print(f"\nValid chunks: {len(valid)}")
        print(f"Rejected chunks: {len(rejected)}")
        for r in rejected:
            print(f"  - Chunk {r['id']}: {r.get('rejection_reason', 'Unknown')}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify-dataset":
        # Verify existing dataset
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/dataset_raw.jsonl"
        
        print(f"Verifying {dataset_path}...")
        
        pairs = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        
        valid, rejected = filter_hallucinated_pairs(pairs)
        
        print(f"\nResults:")
        print(f"  Total pairs: {len(pairs)}")
        print(f"  Valid: {len(valid)}")
        print(f"  Rejected (hallucinations): {len(rejected)}")
        
        if rejected:
            print("\nHallucinated pairs:")
            for i, r in enumerate(rejected[:10]):  # Show first 10
                print(f"  {i+1}. {r['instruction'][:60]}...")
                for issue in r.get('rejection_reason', []):
                    print(f"      └─ {issue}")
    
    else:
        print("Entity Verifier - Hallucination Prevention Layer 3")
        print("Usage:")
        print("  python entity_verifier.py --test            # Run tests")
        print("  python entity_verifier.py --verify-dataset [path]  # Verify dataset")
