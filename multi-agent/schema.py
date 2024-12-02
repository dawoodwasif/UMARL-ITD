# Define Input Schema and Objectives for Each Subtask
tool_specs = {
    "logon": {
        "required_columns": ["user", "date", "pc", "activity", "content"],
        "objective": "Identify suspicious logon activity outside working hours (8 AM to 6 PM) or invalid activity types.",
        "example_input": {
            "user": ["WCR0044", "LRG0155"],
            "date": ["2024-01-02 05:02:50", "2024-01-02 06:33:00"],
            "pc": ["PC-9174", "PC-0450"],
            "activity": ["Logon", "Logoff"],
            "content": ["Login successful", "User logged out"]
        },
        "example_output": "Filtered DataFrame containing rows with suspicious logon activities."
    },
    "file_access": {
        "required_columns": ["user", "file", "access_type", "timestamp"],
        "objective": "Detect file access events classified as suspicious, such as 'delete' operations.",
        "example_input": {
            "user": ["WCR0044", "LRG0155"],
            "file": ["confidential.pdf", "project_plan.docx"],
            "access_type": ["read", "delete"],
            "timestamp": ["2024-01-02 05:15:00", "2024-01-02 06:40:00"]
        },
        "example_output": "Filtered DataFrame with rows flagged for suspicious file access."
    },
    "email": {
        "required_columns": ["user", "email_content", "timestamp"],
        "objective": "Detect suspicious email content containing flagged keywords like 'password', 'urgent', or 'secret'.",
        "example_input": {
            "user": ["WCR0044", "LRG0155"],
            "email_content": ["Request for urgent payment", "Hello!"],
            "timestamp": ["2024-01-02 05:15:00", "2024-01-02 06:40:00"]
        },
        "example_output": "Filtered DataFrame containing rows with suspicious email content."
    },
    "device": {
        "required_columns": ["user", "device", "action", "timestamp"],
        "objective": "Identify unusual device activity, such as connecting unauthorized devices.",
        "example_input": {
            "user": ["WCR0044", "XTR0011"],
            "device": ["USB", "Printer"],
            "action": ["connect", "disconnect"],
            "timestamp": ["2024-01-02 05:15:00", "2024-01-02 06:40:00"]
        },
        "example_output": "Filtered DataFrame containing rows with suspicious device activity."
    },
    "psychometric": {
        "required_columns": ["user", "trait", "value"],
        "objective": "Analyze psychometric traits to detect scores outside valid ranges or suspicious patterns.",
        "example_input": {
            "user": ["WCR0044", "LRG0155"],
            "trait": ["openness", "conscientiousness"],
            "value": [0.8, 1.2]
        },
        "example_output": "Filtered DataFrame highlighting users with psychometric anomalies."
    }
}
