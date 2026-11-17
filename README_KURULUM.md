# DepoNav AI — Dört sınıflandırıcı, zaman CV, Exact TSP ve forklift

**Proje:** Ayakkabı Depolarında Mevsimsel Özelliklerle Ürün Bazlı Talep Sınıflandırması ve Exact TSP ile Sipariş Toplama Rotasının Optimizasyonu.

## Başlangıç — Windows 10/11

1. ZIP içindeki klasörü bilgisayara **tamamen çıkarın**. Eski DepoNav projesinin üzerine yazmayın.
2. **Python 3.11** kurulu olsun (`py -3.11 --version`).
3. Klasörde `Windows_Baslat.bat` çalıştırın. İlk kurulum internet bağlantısı gerektirir.
4. Manuel başlatmak isterseniz PowerShell'de:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`http://localhost:8501` açılır; boşta değilse farklı port kullanılabilir. Terminal açık kalmalıdır.

**Model uyumluluğu:** Modeller **scikit-learn 1.8.0 ve XGBoost 3.1.3** ile kaydedildi; `requirements.txt` bu sürümü sabitler. Anaconda `(base)` yerine `.venv` kullanın. Başka sürümde kayıtlı modeller güvenle yüklenemez.

## Son proje yapısı

- `app.py`: Streamlit'in 6 sayfalı arayüzü — genel bakış, ürün tahmin sıralaması, dört sınıflandırıcı performansı (Accuracy dahil), geçmiş rota ve forklift simülasyonu, yeni rota planı, veri ve metodoloji.
- `DepoNav_Tam_Proje_4_Model_XGBoost_Accuracy_CV.ipynb`: Türkçe açıklamalı **44 hücrelik çalıştırılmış** uçtan uca notebook.
- `train_models.py`: Dört modeli, 3 kronolojik CV fold'u, doğrulama ve bağımsız unseen testi üretir.
- `product_demand.py`: Veri hazırlama, önceki günlerden ürün ve sezon özellikleri.
- `warehouse_core.py`: Exact TSP uygulaması; OR-Tools CP-SAT mevcutsa o, değilse SciPy MILP kesin çözücü.
- `forklift_simulator.py` + `assets/forklift_simulator.html`: Hareketli forklift ve konum/etap sayaçları.
- `models/`: Eğitilmiş **dört** sınıflandırıcı ve doğrulamayla seçilen model paketi.
- `data/`: Customer_Order, Picking_Wave ve Storage_Location CSV dosyaları.
- `outputs/`: CV fold metrikleri, CV OOF tahminleri, ayrı doğrulama, bağımsız test metrikleri/tahminleri, TSP optimum rota dosyaları ve forklift örnek HTML.
- `tests/`: Sekiz otomatik doğrulama testi.

## Görev ve deney kuralları

**Hedef değişken:** Bir sonraki kayıtlı/planlanan operasyon gününde geçmişte görülmüş bir ürün referansı sipariş edilir mi? İkili sınıflandırma. Ürün adedi, gerçek stok seviyesi ve gerçek forklift sürüş süresi değildir.

**Sezon özellikleri:** Ay, çeyrek, haftanın günü, yıl ve ayın sinüs/kosinüs dönüşümleri, geçmişte aynı sezondaki talep sıklığı ve geçmiş davranış özellikleri. **Dört modelin hepsinde aynı özellikler vardır.** Sezonlu–sezonsuz karşılaştırması **kaldırılmıştır**. Referans kodundan bot/terlik türü çıkarılmaz; Ocak–Ekim 2023 dönemi tek yıl olduğu için yıllık sezon etkisi kanıtlanmış değildir.

**Dört sınıflandırıcı:** Logistic Regression (baseline), Decision Tree, Random Forest ve XGBoost. 30 günlük talep sıklığı ayrıca **kurala dayalı bir referanstır; beşinci ML modeli değildir**.

**Kronolojik bölme:** 178 kayıtlı operasyondan ilk 30 gün özellik geçmişi. Sonraki 96 gün eğitim, 26 gün ayrı doğrulama, **son 26 gün (2023-09-06–2023-10-19)** bağımsız unseen test. Veri setinde kaydı olmayan takvim günleri otomatik sıfır kabul edilmez. Test içindeki sonraki günlerin özellikleri yalnız kendi tarihinden önceki gözlemlerden üretilir (günlük walk-forward).

**Zaman CV:** Yalnız eğitim döneminde üç gelişen pencere: sırasıyla 36/56/76 eğitim günü ve 20 günlük fold doğrulaması. Hiperparametre CV ortalama PR-AUC ile, karar eşikleri CV dış-kat tahminlerinde (OOF) **Precision≥0,60 hedefi altında maksimum F2** ile belirlenir. Ayrı doğrulama seti F2 (+precision hedefi) ile son model ailesini seçer. **Görülmemiş testin skoruna göre model/eşik seçilmez.**

**Accuracy ve Precision–Recall kararı:** Accuracy ve Balanced Accuracy ek olarak raporlanır; yanlış negatifleri gizleyebileceğinden tek başına model seçimini belirlemez.

**Precision–Recall kararı:** Eksik talep tahmini (FN), ürün hazırlığını kaçırabileceğinden recall önceliklidir; yalancı alarm (FP) gereksiz hazırlık doğurur. Bu nedenle **F2**, precision hedefi ve kapasiteye göre **Precision@20 / Recall@20** birlikte yorumlanır. Precision tabanı testte garanti değildir.

## Gerçek sonuçlar — bağımsız unseen test

Son **26 operasyon günü** (2023-09-06–2023-10-19), toplam **5.342** SKU-gün adayı. Modellerin hepsi sezon özelliklerini kullanır.

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F2 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (Baseline) | 0.747 | 0.739 | 0.615 | 0.714 | 0.691 | 0.669 |
| Decision Tree | 0.679 | 0.715 | 0.522 | 0.829 | 0.742 | 0.662 |
| Random Forest | 0.704 | 0.743 | 0.545 | 0.867 | 0.775 | 0.660 |
| XGBoost | 0.696 | 0.739 | 0.536 | 0.877 | 0.778 | 0.672 |

**Seçilen ana model: Random Forest.** Bu model **unseen testten önce**, yalnız eğitim içi CV ve ayrı doğrulamada precision tabanı altında F2'ye göre seçildi. XGBoost'un unseen PR-AUC ve Recall avantajı test sonuçlarını gördükten sonra model seçimini geriye dönük değiştirme gerekçesi değildir.

**İş açısından öncelik:** Yanlış negatifler (FN) talebi kaçırma riski doğurur; bu nedenle recall ve F2 önemlidir. Precision düşükse gereksiz hazırlık artar. Testte Random Forest'ın precision'ı yaklaşık **%54,5** kaldı ve önceden belirlenen **%60 hedefine ulaşamadı**. Baseline Logistic Regression'ın Accuracy'si daha yüksek olsa da Recall'ı daha düşük; tek başına Accuracy model seçmek için doğru değil. **Kurala dayalı 30 günlük sıklık** ilave bir sınıflandırıcı değil, bağımsız referanstır.

Bu sonuçlar gelecekteki gerçek 2026 stok talebinin doğrulanmış tahminleri değildir. Yeni ürünlerin görünmeme (cold-start) problemi vardır. Gözlem kapsamı geçmişte bilinen SKU'larla sınırlıdır.

## Rota + forklift sınırlaması

Önceki uygulamadaki bütün geçmiş optimum rotalar, yeni raf seçerek Exact TSP çalıştırma ve şematik hareketli forklift simülasyonu **korundu**. Hesaplanan mesafelerin doğruluğu, yaklaşık koridor bağlantıları ve varsayılan giriş/çıkış noktasına bağlıdır. `OPTIMAL`, yalnız tanımlanan matematiksel model için minimum demektir; gerçek depoda ölçülmüş metre garantisi vermez.

## Yeniden üretim

```powershell
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests -q
```

Notebook için:

```powershell
.\.venv\Scripts\python.exe -m notebook DepoNav_Tam_Proje_4_Model_XGBoost_Accuracy_CV.ipynb
```

**Test durumu:** 8 otomatik Python testi başarılı. Notebook'un kod hücreleri Jupyter çekirdeği dışında aynı Python ortamında sırasıyla çalıştırılarak çıktıları kaydedildi. Web arayüzünün tüm Streamlit sayfalarının gerçek Windows/tarayıcı ortamında açıldığı burada doğrulanamadı.
