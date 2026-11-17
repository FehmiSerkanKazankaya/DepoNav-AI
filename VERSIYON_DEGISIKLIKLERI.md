# Final sürüm değişiklikleri

- Sınıflandırıcılar: Logistic Regression (baseline), Decision Tree, Random Forest, XGBoost.
- Her modelde aynı mevsim ve gecikmeli talep özellikleri.
- Accuracy, Balanced Accuracy, Precision, Recall, F1, F2, ROC-AUC, PR-AUC, Top-20 metrikleri.
- Zaman sıralı 3 fold çapraz doğrulama + ayrı doğrulama + tamamen ayrılmış son 26 gün unseen test.
- Model seçimi teste bakılmadan yapıldı; precision tabanı ve recall ağırlıklı F2 dikkate alındı.
- Streamlit uygulamasında dört model seçimi, Accuracy kartı ve karşılaştırma grafiği.
- Kesin TSP optimizasyonu, geçmiş rotalar ve 2B forklift simülasyonu korundu.
- requirements.txt: scikit-learn 1.8.0 + xgboost 3.1.3; Windows için izole venv önerilir.

**Sınırlamalar:** 2023 Ocak–Ekim aralığı tek kısmi yıl; sezonluk etkiler tekrarlayan yıllarla doğrulanmadı. Rotalar fiziksel koridorlarla tam doğrulanmamış model üzerinde hesaplanır. Final sürümde seçilen Random Forest testte precision hedefini sağlayamadı.
