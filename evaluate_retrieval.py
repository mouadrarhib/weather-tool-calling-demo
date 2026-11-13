from filtered_semantic_search import retrieve_passage

test_cases = [
    {
        "question": "How can I cancel my booking?",
        "location": "Casablanca",
        "expected_id": "policy-01",
        "answer_supported": True,
    },
    {
        "question": "Where can I cancel my reservation?",
        "location": "Casablanca",
        "expected_id": "policy-01",
        "answer_supported": True,
    },
    {
        "question": "Do your vehicles receive regular maintenance?",
        "location": "Casablanca",
        "expected_id": "policy-04",
        "answer_supported": True,
    },
    {
        "question": "How can I cancel my booking?",
        "location": "Rabat",
        "expected_id": None,
        "answer_supported": False,
    },
    {
        "question": "What is the cancellation fee?",
        "location": "Casablanca",
        "expected_id": "policy-01",
        "answer_supported": False,
    },
    {
        "question": "What is the capital of Japan?",
        "location": "Casablanca",
        "expected_id": None,
        "answer_supported": False,
    },
    {
        "question": "I need to cancel my reservation. Where do I do that?",
        "location": "Casablanca",
        "expected_id": "policy-01",
        "answer_supported": True,
    },
    {
        "question": "Are the cars maintained regularly?",
        "location": "Casablanca",
        "expected_id": "policy-04",
        "answer_supported": True,
    },
    {
        "question": "How do I bake a chocolate cake?",
        "location": "Casablanca",
        "expected_id": None,
        "answer_supported": False,
    },
]

passed = 0

for case in test_cases:
    passage = retrieve_passage(
        case["question"],
        location=case["location"],
    )

    actual_id = passage["id"] if passage is not None else None
    matches = actual_id == case["expected_id"]

    if matches:
        passed += 1

    print("\nQuestion:", case["question"])
    print("Location:", case["location"])
    print("Expected:", case["expected_id"])
    print("Retrieved:", actual_id)
    print("Result:", "PASS" if matches else "FAIL")

    print("Answer supported by sample documents:", case["answer_supported"])

    if passage is not None:
        print("Similarity:", round(passage["similarity"], 3))

print(f"\nPassed: {passed}/{len(test_cases)}")