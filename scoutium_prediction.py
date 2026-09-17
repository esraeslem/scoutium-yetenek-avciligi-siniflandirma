##################################################################################
# MAKİNE ÖĞRENMESİ İLE YETENEK AVCILIĞI SINIFLANDIRMA (SCOUTIUM)
##################################################################################

# İş Problemi
# -----------
# Scout'lar tarafından izlenen futbolcuların, maç içerisinde gözlemlenen özelliklerine
# verilen puanlara göre; oyuncunun "average" (ortalama) mı yoksa "highlighted"
# (öne çıkan/yetenekli) sınıfına mı ait olduğunu tahmin eden bir sınıflandırma modeli
# geliştirmek.
#
# Veri Seti Hikayesi
# -------------------
# Veri seti Scoutium'dan; maçlarda gözlemlenen futbolcuların özelliklerine göre
# scout'ların değerlendirdiği oyunculara ait, maç içerisinde puanlanan özellik ve
# puan bilgilerinden oluşmaktadır.
#
# scoutium_attributes.csv (8 değişken, 10.730 gözlem)
#   - task_response_id : Bir scout'un bir maçtaki bir takım kadrosuna dair değerlendirme kümesi
#   - match_id         : İlgili maçın id'si
#   - evaluator_id     : Değerlendiricinin (scout'un) id'si
#   - player_id        : İlgili oyuncunun id'si
#   - position_id      : Oyuncunun o maçta oynadığı pozisyonun id'si
#                         (1: Kaleci, 2: Stoper, 3: Sağ bek, 4: Sol bek,
#                          5: Defansif orta saha, 6: Merkez orta saha, 7: Sağ kanat,
#                          8: Sol kanat, 9: Ofansif orta saha, 10: Forvet)
#   - analysis_id      : Bir scout'un bir maçta bir oyuncuya dair özellik değerlendirme kümesi
#   - attribute_id     : Oyuncunun değerlendirildiği her bir özelliğin id'si
#   - attribute_value  : Scout'un oyuncunun bir özelliğine verdiği puan
#
# scoutium_potential_labels.csv (5 değişken, 322 gözlem)
#   - task_response_id, match_id, evaluator_id, player_id : (yukarıdaki ile aynı)
#   - potential_label  : Scout'un oyuncu ile ilgili nihai kararı (HEDEF DEĞİŞKEN)
##################################################################################

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
warnings.filterwarnings("ignore")

ATTRIBUTES_PATH = "datasets/scoutium_attributes.csv"
LABELS_PATH = "datasets/scoutium_potential_labels.csv"
IMPORTANCE_PLOT_PATH = "outputs/feature_importance.png"


##################################################################################
# ADIM 1: Veri setlerini okutunuz.
##################################################################################

def load_data(attributes_path=ATTRIBUTES_PATH, labels_path=LABELS_PATH):
    attributes = pd.read_csv(attributes_path, sep=";")
    potential_labels = pd.read_csv(labels_path, sep=";")
    return attributes, potential_labels


##################################################################################
# ADIM 2: İki veri setini "task_response_id", "match_id", "evaluator_id",
#         "player_id" değişkenleri üzerinden merge fonksiyonu ile birleştiriniz.
##################################################################################

def merge_data(attributes, potential_labels):
    df = attributes.merge(
        potential_labels,
        on=["task_response_id", "match_id", "evaluator_id", "player_id"],
        how="left",
    )
    return df


##################################################################################
# ADIM 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
##################################################################################

def drop_goalkeepers(df):
    return df[df["position_id"] != 1]


##################################################################################
# ADIM 4: potential_label içerisindeki below_average sınıfını kaldırınız.
#         (below_average sınıfı tüm veri setinin sadece %1'ini oluşturur.)
##################################################################################

def drop_below_average(df):
    return df[df["potential_label"] != "below_average"]


##################################################################################
# ADIM 5: Pivot table oluşturunuz.
#   5.1: İndekste "player_id", "position_id", "potential_label";
#        sütunlarda "attribute_id"; değerlerde "attribute_value" olacak şekilde.
#   5.2: reset_index ile indeksleri değişkene çeviriniz ve "attribute_id"
#        sütun isimlerini stringe çeviriniz.
##################################################################################

def create_pivot_table(df):
    pivot_df = pd.pivot_table(
        df,
        values="attribute_value",
        index=["player_id", "position_id", "potential_label"],
        columns="attribute_id",
    )
    pivot_df = pivot_df.reset_index()
    pivot_df.columns = pivot_df.columns.map(str)
    return pivot_df


##################################################################################
# ADIM 6: LabelEncoder ile "potential_label" kategorilerini (average, highlighted)
#         sayısal olarak ifade ediniz.
##################################################################################

def encode_target(pivot_df):
    labelencoder = LabelEncoder()
    pivot_df["potential_label"] = labelencoder.fit_transform(pivot_df["potential_label"])
    # average -> 0, highlighted -> 1
    return pivot_df, labelencoder


##################################################################################
# ADIM 7 & 8: Sayısal değişken kolonlarını "num_cols" adında bir listeye atayınız
#             ve StandardScaler ile ölçeklendiriniz.
##################################################################################

def scale_numeric_columns(pivot_df):
    num_cols = pivot_df.columns[3:]  # player_id, position_id, potential_label hariç tüm attribute kolonları
    scaler = StandardScaler()
    pivot_df[num_cols] = scaler.fit_transform(pivot_df[num_cols])
    return pivot_df, num_cols


##################################################################################
# ADIM 9: Minimum hata ile futbolcuların potansiyel etiketini tahmin eden bir
#         makine öğrenmesi modeli geliştiriniz.
#         (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
##################################################################################

def build_model(pivot_df):
    y = pivot_df["potential_label"]
    X = pivot_df.drop(["potential_label", "player_id"], axis=1)

    lgbm_model = LGBMClassifier(random_state=17, verbose=-1)

    cv_results = cross_validate(
        lgbm_model,
        X,
        y,
        cv=5,
        scoring=["roc_auc", "f1", "precision", "recall", "accuracy"],
    )

    print("########## Model Başarı Metrikleri (5-Fold CV Ortalaması) ##########")
    print(f"ROC_AUC   : {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"F1        : {round(cv_results['test_f1'].mean(), 4)}")
    print(f"Precision : {round(cv_results['test_precision'].mean(), 4)}")
    print(f"Recall    : {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Accuracy  : {round(cv_results['test_accuracy'].mean(), 4)}")

    lgbm_model.fit(X, y)
    return lgbm_model, X, y, cv_results


##################################################################################
# ADIM 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu
#          kullanarak özelliklerin sıralamasını çizdiriniz.
##################################################################################

def plot_importance(model, features, num=15, save_path=IMPORTANCE_PLOT_PATH):
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": features.columns}
    )
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title("Değişken Önem Düzeyleri (Feature Importance)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Feature importance grafiği kaydedildi: {save_path}")
    plt.show()


##################################################################################
# ANA AKIŞ (MAIN PIPELINE)
##################################################################################

def main():
    attributes, potential_labels = load_data()

    df = merge_data(attributes, potential_labels)
    df = drop_goalkeepers(df)
    df = drop_below_average(df)

    pivot_df = create_pivot_table(df)
    pivot_df, labelencoder = encode_target(pivot_df)
    pivot_df, num_cols = scale_numeric_columns(pivot_df)

    model, X, y, cv_results = build_model(pivot_df)
    plot_importance(model, X)

    return model, X, y, cv_results


if __name__ == "__main__":
    main()
