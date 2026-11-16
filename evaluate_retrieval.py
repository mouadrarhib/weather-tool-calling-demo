from filtered_semantic_search import retrieve_passage

test_cases = [
    {
        "question": "How can I cancel my booking?",
        "location": "Casablanca",
        "expected_id": "rental-pdf-01",
        "expected_page": 1,
        "answer_supported": True,
    },
    {
        "question": "Where can I cancel my reservation?",
        "location": "Casablanca",
        "expected_id": "rental-pdf-01",
        "expected_page": 1,
        "answer_supported": True,
    },
    {
        "question": "Do your vehicles receive regular maintenance?",
        "location": "Casablanca",
        "expected_id": "rental-pdf-02",
        "expected_page": 2,
        "answer_supported": True,
    },
    {
        "question": "How can I cancel my booking?",
        "location": "Rabat",
        "expected_id": None,
        "expected_page": None,
        "answer_supported": False,
    },
    {
        "question": "What is the cancellation fee?",
        "location": "Casablanca",
        "expected_id": "rental-pdf-01",
        "expected_page": 1,
        "answer_supported": True,
    },
    {
        "question": "What is the capital of Japan?",
        "location": "Casablanca",
        "expected_id": None,
        "expected_page": None,
        "answer_supported": False,
    },
    {
        "question": "I need to cancel my reservation. Where do I do that?",
        "location": "Casablanca",
        "expected_id": "rental-pdf-01",
        "expected_page": 1,
        "answer_supported": True,
    },
    {
        "question": "Are the cars maintained regularly?",
        "location": "Casablanca",
        "expected_id": "rental-pdf-02",
        "expected_page": 2,
        "answer_supported": True,
    },
    {
        "question": "How do I bake a chocolate cake?",
        "location": "Casablanca",
        "expected_id": None,
        "expected_page": None,
        "answer_supported": False,
    },
    {
        "question": "Is the cancellation fee refundable?",
        "location": "Casablanca",
        "expected_id": "rental-pdf-01",
        "expected_page": 1,
        "answer_supported": False,
    },
    {
        "question": "What documents should I bring when picking up the car?",
        "location": "Casablanca",
        "expected_id": "rental-pdf-03",
        "expected_page": 3,
        "answer_supported": True,
    },
]


passed = 0

for case in test_cases:
    passage = retrieve_passage(
        case["question"],
        location=case["location"],
    )

    actual_id = passage["id"] if passage is not None else None
    actual_page = passage["page"] if passage is not None else None

    matches = (
        actual_id == case["expected_id"]
        and actual_page == case["expected_page"]
    )

    if matches:
        passed += 1

    print("\nQuestion:", case["question"])
    print("Location:", case["location"])
    print("Expected ID:", case["expected_id"])
    print("Retrieved ID:", actual_id)
    print("Expected page:", case["expected_page"])
    print("Retrieved page:", actual_page)
    print("Result:", "PASS" if matches else "FAIL")
    print("Answer supported by sample documents:", case["answer_supported"])

    if passage is not None:
        print("Similarity:", round(passage["similarity"], 3))

print(f"\nPassed: {passed}/{len(test_cases)}")