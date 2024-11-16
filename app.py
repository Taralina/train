import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io

# Загрузка предобученной модели
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 11)  # Здесь должно быть количество классов, которое у вас есть
model.load_state_dict(torch.load("model_resnet18.pth", map_location=torch.device('cpu')))
model.eval()

# Определение преобразований для изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Функция для обработки изображения и предсказания
def predict_image(image):
    # Применение преобразований к изображению
    image = transform(image).unsqueeze(0)
    
    # Передаем изображение в модель
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

# Интерфейс Streamlit
st.title("Прогнозирование классов изображений с помощью модели ResNet18")
st.write("Загрузите изображение для классификации.")

# Загружаем изображение
uploaded_image = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Открытие изображения
    image = Image.open(uploaded_image)
    
    # Отображаем изображение
    st.image(image, caption="Загруженное изображение", use_column_width=True)
    
    # Прогноз
    if st.button("Предсказать класс"):
        label = predict_image(image)
        st.write(f"Предсказанный класс: {label}")