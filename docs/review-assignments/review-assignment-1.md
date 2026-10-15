# Task

- Review a participant's feedback based on the material contained in the current repo and, specifically, the rubrics under "02_activities/assignments/assignment_1_rubric_clean.xlsx".
- The review should be didactic and constructive, we are giving feedback to enhance learner's experience and not to do a deployment in production.
- When you give feedback, point to the course resources, especially the jupyter notebooks contained in 01_materials\labs and the code in 05_src.
- The relevant lessons for assignment 1 are session 02_data_engineering. 
- Post the feedback as a comment to the PR.

# How to Give High-Quality Feedback

Effective feedback helps participants build real world coding and collaboration skills. Use comments in GitHub pull requests to encourage reflection, improve code quality, and clarify expectations, especially for those marked as missing or incomplete, so they can learn from their mistakes and improve.
Best Practices

- Be respectful and suggest improvements, not just problems.
- Keep it clear avoid jargon and align with module terminology.
- Focus on the code, calling out both strengths and fixes.
- Be timely so participants have time to revise.
- Stay organized with bullet points and line level comments.
- Use GitHub tools well: inline comments, review labels, and clear PR descriptions.
- Follow up to confirm changes were made.

- Note: Think like a senior developer reviewing a junior’s pull request, your job is to guide and grow their skills.

## Bad Feedback Example

- "This code needs to be improved. Your model isn’t right."
- "The naming convention here is weird. Fix it."
- "Your RMSPE is too high."
- "A1: FAIL"

Why it's bad: Vague, negative, unhelpful.

## Good Feedback Example

- "For Question 3, try using train_test_split from sklearn for better reproducibility. See: [link]."
- "Great job initializing LinearRegression. But you should encode origin, it’s non-numeric."
- "The RMSE is high, check for multicollinearity or consider adjusting your predictors."

Why it's good: Specific, actionable, and encouraging.