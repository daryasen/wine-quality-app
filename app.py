from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics

ROOT = Path(__file__).parent
st.set_page_config(page_title='Подбор вина', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    # Загружаю модель один раз на сервер, чтобы не читать её при каждом движении ползунка
    return joblib.load(ROOT / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(ROOT / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    # Считаю вероятности один раз, потому что они не зависят от порога
    return _model.predict_proba(X)[:, 1]


def read_wines(upload, columns):
    data = pd.read_csv(upload)
    missing = []
    for name in columns:
        if name not in data.columns:
            missing.append(name)
    if missing:
        raise ValueError('В файле не хватает столбцов: ' + ', '.join(missing))
    if data.empty:
        raise ValueError('В файле нет строк с винами')
    X = data[columns].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Уберите пропуски и бесконечные значения из признаков')
    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red должны быть только 0 и 1')
    return data, X


model = load_model()
test = load_test()
X_test = test.drop(columns='target')
y_test = test['target']
p_test = test_proba(model, X_test)
config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
threshold = st.sidebar.slider('Порог хорошего вина', 0.0, 1.0,
                              float(config['threshold']), step=0.001, format='%.3f')
st.sidebar.caption('Один порог используется во всех вкладках')
st.title('Подбор хороших вин')
st.write('Модель оценивает вино по результатам лабораторного анализа')
one, quality, batch, weights = st.tabs(['Одно вино', 'Модельные метрики', 'Загрузка CSV', 'Что выучила модель'])

with one:
    st.subheader('Характеристики вина')
    labels = {'fixed_acidity': 'Фиксированная кислотность',
              'volatile_acidity': 'Летучая кислотность', 'citric_acid': 'Лимонная кислота',
              'residual_sugar': 'Остаточный сахар', 'chlorides': 'Хлориды',
              'free_sulfur_dioxide': 'Свободный диоксид серы',
              'total_sulfur_dioxide': 'Общий диоксид серы', 'density': 'Плотность',
              'pH': 'pH', 'sulphates': 'Сульфаты', 'alcohol': 'Алкоголь'}
    values = {}
    columns = st.columns(3)
    for i, name in enumerate(X_test.columns):
        with columns[i % 3]:
            if name == 'is_red':
                values[name] = int(st.toggle('Красное вино', value=bool(X_test[name].median())))
            else:
                values[name] = st.number_input(labels[name],
                    min_value=float(X_test[name].min()), max_value=float(X_test[name].max()),
                    value=float(X_test[name].median()), format='%.5f', key=name)
    wine = pd.DataFrame([values])[X_test.columns]
    probability = model.predict_proba(wine)[0, 1]
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.success('Добавить в подборку')
    else:
        st.info('Пропустить это вино')

with quality:
    st.subheader('Качество на отложенной выборке')
    pred = p_test >= threshold
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred)
    a, b, c = st.columns(3)
    a.metric('Precision', f'{precision:.3f}')
    b.metric('Recall', f'{recall:.3f}')
    c.metric('F1', f'{f1:.3f}')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    matrix = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    metrics.ConfusionMatrixDisplay(matrix, display_labels=['Обычное', 'Хорошее']).plot(ax=axes[0], colorbar=False)
    metrics.RocCurveDisplay.from_predictions(y_test, p_test, ax=axes[1])
    tn, fp, fn, tp = matrix.ravel()
    axes[1].scatter(fp / (fp + tn), recall, color='red', label='Текущий порог')
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].legend()
    metrics.PrecisionRecallDisplay.from_predictions(y_test, p_test, ax=axes[2])
    axes[2].scatter(recall, precision, color='red', label='Текущий порог')
    axes[2].axhline(y_test.mean(), linestyle='--', color='gray', label='Случайная модель')
    axes[2].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('При увеличении порога подборка становится строже и обычно короче, но часть хороших вин теряется')
    st.write('При уменьшении порога хороших вин находится больше, но в подборку чаще попадают обычные вина')

with batch:
    st.subheader('Оценка нескольких вин')
    st.write('Загрузите CSV с теми же 12 признаками, что и в исходных данных')
    st.download_button('Скачать пример CSV', X_test.head(5).to_csv(index=False).encode('utf-8'),
                       file_name='wine_example.csv', mime='text/csv')
    upload = st.file_uploader('CSV с характеристиками вин', type=['csv'])
    if upload is not None:
        try:
            data, features = read_wines(upload, list(X_test.columns))
            data['probability'] = model.predict_proba(features)[:, 1]
            data['prediction'] = (data['probability'] >= threshold).astype(int)
            st.dataframe(data, hide_index=True)
            st.download_button('Скачать результат', data.to_csv(index=False).encode('utf-8-sig'),
                               file_name='wine_predictions.csv', mime='text/csv')
        except (ValueError, TypeError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error('Не удалось обработать CSV: ' + str(error))

with weights:
    st.subheader('Вклад признаков')
    # Беру коэффициенты и сортирую их от меньшего к большему
    coefficients = pd.Series(model[-1].coef_[0], index=X_test.columns)
    coefficients = coefficients.sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    # Показываю направление влияния по положению относительно нуля
    ax.barh(coefficients.index, coefficients.values, color='steelblue')
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Коэффициент после масштабирования')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего за хорошее вино:', ', '.join(coefficients.tail(3).index[::-1]))
    st.write('Сильнее всего против:', ', '.join(coefficients.head(3).index))
    st.caption('Коэффициенты показывают связь при прочих равных, причинный эффект из них не следует')
