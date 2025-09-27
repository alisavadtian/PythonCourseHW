import streamlit as st
import pandas as pd
import joblib

# Загружаем модель
model = joblib.load(r'D:\downloads\train_data\rf_model.pkl')

st.title("Прогноз цены недвижимости")

# Ввод признаков
postcode = st.number_input("Postcode", min_value=101000.0, max_value=143989.0, value=143989.0)
rooms = st.number_input("Количество комнат", min_value=1.0, max_value=12.0, value=3.0)
total_square = st.number_input("Общая площадь (м²)", min_value=8.2, max_value=1157.0, value=1157.0)

# Кнопка для предсказания
if st.button("Сделать прогноз"):
    # Формируем DataFrame для модели
    input_df = pd.DataFrame({
        "postcode": [postcode],
        "rooms": [rooms],
        "total_square": [total_square]
    })

    # Делаем предсказание
    prediction = model.predict(input_df)[0]

    # Показываем результат
    st.success(f"Прогнозируемая цена: {prediction:,.2f} руб.")