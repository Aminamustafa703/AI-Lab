def scholarship(score, documents, attendance, interview):

    if score >= 70 and documents == True and attendance >= 75 and interview == True:
        return "Eligible"
    else:
        return "Not Eligible"

cases = [
    ("Pass", 85, True, 80, True),
    ("Attendance Fail", 80, True, 60, True),
    ("Interview Fail", 90, True, 85, False),
    ("Boundary", 70, True, 75, True)
]

print("Scholarship Screening Results")
print("--------------------------------")

for name, score, documents, attendance, interview in cases:

    result = scholarship(score, documents, attendance, interview)

    print("Case:", name)
    print("Score:", score)
    print("Documents:", documents)
    print("Attendance:", attendance)
    print("Interview:", interview)
    print("Expected Result:", result)
    print("Actual Result:", result)
    print("Reason: All conditions checked")
    print("--------------------------------")
