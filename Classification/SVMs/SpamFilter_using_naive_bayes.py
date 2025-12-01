import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

emails = [
    "Buy cheap viagra now",          # Spam
    "Meeting at 3 pm tomorrow",      # Not Spam
    "Get rich quick scheme",         # Spam
    "Can we have lunch today?",      # Not Spam
    "Discount on rolex watches",     # Spam
    "Project updates attached",      # Not Spam
    "Application Received – e& Egypt", # Not Spam
    "Team Vibe Take-Home Tests🚀"     # Not Spam
]
labels = [1, 0, 1, 0, 1, 0, 0, 0] # تم تحديث الـ labels لـ 8 رسائل

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
print("القاموس الجديد:", vectorizer.get_feature_names_out())

clf = MultinomialNB(alpha=1.0)
clf.fit(X, labels)
new_emails = ["Meeting at 3 pm tomorrow", "Can we have lunch today?", "Discount on rolex watches", "Project updates attached" ,
"Buy cheap viagra now", "Get rich quick scheme", "Discount on rolex watches", "Project updates attached" , 
"Application Received – e& Egypt" , "Team Vibe Take-Home Tests🚀"]
prediction = clf.predict(vectorizer.transform(new_emails))

print("\n--- النتائج المُحسَّنة ---")
for i in range(len(new_emails)):
    if prediction[i] == 1:
        print(f"Email: '{new_emails[i]}' IS SPAM 🚨")
    else:
        print(f"Email: '{new_emails[i]}' IS NOT SPAM ✅")

proba = clf.predict_proba(vectorizer.transform(new_emails))
print(f"confidence of the model: {proba}")