# app.py

# Bibliotecas Necessárias:
# pip install streamlit tensorflow scikit-learn Pillow pandas

import streamlit as st  # Framework Streamlit para criação da interface web.
import tensorflow as tf  # Biblioteca TensorFlow para carregar e usar o modelo CNN.
from tensorflow.keras.preprocessing import image  # Para pré-processar a imagem antes de passar para o modelo.
import numpy as np  # Para operações numéricas com arrays.
from PIL import Image  # Para manipulação de imagens.
import pandas as pd  # Para criar e salvar o histórico de previsões em um arquivo CSV.
import datetime  # Para registrar a data e hora das previsões.
import os  # Para lidar com caminhos de arquivos.

# Função para carregar o modelo CNN
@st.cache_resource  # Garante que o modelo seja carregado apenas uma vez para otimizar o desempenho.
def load_model():
    """Carrega o modelo CNN treinado."""
    try:
        model = tf.keras.models.load_model('VGG16.h5')  # Carrega o modelo a partir do arquivo VGG16.h5.
        print("Modelo carregado com sucesso!")  # Confirma que o modelo foi carregado.
        return model  # Retorna o modelo carregado.
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")  # Exibe uma mensagem de erro se o modelo não puder ser carregado.
        return None  # Retorna None para indicar que o modelo não foi carregado corretamente.

# Função para pré-processar a imagem
def preprocess_image(img):
    """Pré-processa a imagem para o formato esperado pelo modelo."""
    try:
        img = img.resize((224, 224))  # Redimensiona a imagem para 224x224 pixels.
        img_array = image.img_to_array(img)  # Converte a imagem em um array NumPy.
        img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão extra para representar o lote (batch).
        img_array /= 255.  # Normaliza os valores dos pixels para o intervalo [0, 1].
        return img_array  # Retorna a imagem pré-processada.
    except Exception as e:
        st.error(f"Erro ao pré-processar a imagem: {e}") # Exibe erro caso ocorra um problema ao pré-processar
        return None

# Função para fazer a previsão
def predict(model, img_array):
    """Realiza a previsão da classe da imagem usando o modelo."""
    try:
        prediction = model.predict(img_array)  # Faz a previsão usando o modelo.
        predicted_class_index = np.argmax(prediction)  # Obtém o índice da classe com maior probabilidade.
        confidence = prediction[0][predicted_class_index]  # Obtém a probabilidade da classe predita.

        # Adiciona uma verificação de confiança mínima
        if confidence < 0.5:  # Define um limiar de confiança (ajuste conforme necessário).
            return None, confidence  # Retorna None se a confiança for baixa.
        else:
            return predicted_class_index, confidence  # Retorna o índice da classe e a probabilidade.
    except Exception as e:
        st.error(f"Erro ao prever a classe da imagem: {e}")  # Exibe um erro se houver um problema na previsão.
        return None, 0  # Retorna None e probabilidade 0 em caso de erro.

# Função para obter informações sobre a doença
def get_disease_info(class_name):
    """Retorna informações sobre a doença com base na classe prevista."""
    disease_info = {  # Dicionário com informações sobre cada classe.
        "saudavel": "A folha está saudável e não apresenta sinais de doença.",
        "doenca_1": "Doença 1 causa manchas nas folhas. Medidas de controle incluem fungicidas.",
        "doenca_2": "Doença 2 afeta o crescimento da planta. Recomenda-se irrigação adequada e remoção de folhas infectadas.",
        "doenca_3": "Doença 3 provoca a descoloração das folhas. A aplicação de fertilizantes pode ajudar.",
        "doenca_4": "Doença 4 leva ao apodrecimento das folhas. O controle biológico pode ser uma solução.",
    }
    return disease_info.get(class_name, "Informação não disponível.")  # Retorna a informação ou uma mensagem padrão se a classe não for encontrada.

# Função para salvar o histórico de previsões
def save_prediction_history(filename, predicted_class, confidence):
    """Salva a previsão no arquivo CSV de histórico."""
    try:
        now = datetime.datetime.now()  # Obtém a data e hora atuais.
        data = {'filename': [filename],  # Nome do arquivo da imagem.
                'datetime': [now.strftime("%Y-%m-%d %H:%M:%S")],  # Data e hora formatadas.
                'predicted_class': [predicted_class],  # Classe prevista.
                'confidence': [confidence]}  # Probabilidade da previsão.
        df = pd.DataFrame(data)  # Cria um DataFrame com os dados.
        if os.path.exists('prediction_history.csv'):  # Verifica se o arquivo de histórico já existe.
            df_existing = pd.read_csv('prediction_history.csv')  # Lê o arquivo existente.
            df = pd.concat([df_existing, df], ignore_index=True)  # Concatena os dados novos com os existentes.
        df.to_csv('prediction_history.csv', index=False)  # Salva o DataFrame no arquivo CSV.
    except Exception as e:
        st.error(f"Erro ao salvar o histórico de predições: {e}") # Emite um erro se houver falha ao salvar o histórico

# Configuração da página Streamlit
st.set_page_config(
    page_title="Diagnóstico de Doenças em Cana-de-Açúcar",  # Título da página.
    page_icon="🌱",  # Ícone da página.
    layout="centered",  # Layout da página.
    initial_sidebar_state="expanded",  # Estado inicial da barra lateral.
)

# Customização do tema
st.markdown(
    """
    <style>
        body {
            color: #000000;
            background-color: #FFFFFF;
        }
        .streamlit-expanderHeader {
            font-weight: bold;
            color: #000000;
        }
    </style>
    """,
    unsafe_allow_html=True,  # Permite usar HTML no Markdown.
)

# Barra lateral
with st.sidebar:  # Define a barra lateral.
    st.image("logo.png", width=150)  # Adiciona o logotipo à barra lateral.
    st.title("Diagnóstico de Doenças")  # Adiciona o título à barra lateral.
    st.markdown("Carregue uma imagem da folha para identificar possíveis doenças.")  # Adiciona uma descrição.

# Carrega o modelo
model = load_model()  # Carrega o modelo CNN.

# Título principal do dashboard
st.title("Diagnóstico de Doenças em Cana-de-Açúcar")  # Título do dashboard.

# Uploader de imagem
uploaded_file = st.file_uploader("Carregue uma imagem da folha...", type=["jpg", "jpeg", "png"])  # Cria o uploader de arquivos.

# Processamento da imagem carregada
if uploaded_file is not None:  # Verifica se um arquivo foi carregado.
    try:
        img = Image.open(uploaded_file)  # Abre a imagem usando o Pillow.
        st.image(img, caption="Imagem Carregada", use_column_width=True)  # Exibe a imagem no dashboard.

        img_array = preprocess_image(img)  # Pré-processa a imagem.

        # Garante que a imagem foi corretamente pré-processada
        if img_array is not None:

            # Verifica se o modelo foi carregado corretamente
            if model is not None:
                predicted_class_index, confidence = predict(model, img_array)  # Faz a previsão.

                if predicted_class_index is not None:
                    class_names = ['saudavel', 'doenca_1', 'doenca_2', 'doenca_3', 'doenca_4']  # Nomes das classes.
                    predicted_class_name = class_names[predicted_class_index]  # Obtém o nome da classe predita.

                    st.write("## Resultado da Classificação:")  # Título do resultado.
                    st.write(f"Doença Detectada: {predicted_class_name.replace('_', ' ').title()}")  # Exibe o nome da doença.
                    st.write(f"Probabilidade: {confidence:.2f}")  # Exibe a probabilidade.

                    disease_info = get_disease_info(predicted_class_name)  # Obtém informações sobre a doença.

                    with st.expander("Informações sobre a Doença"):  # Cria um expander para exibir as informações.
                        st.write(disease_info)  # Exibe as informações sobre a doença.

                    save_prediction_history(uploaded_file.name, predicted_class_name, confidence)  # Salva a previsão no histórico.
                else:
                    st.warning("Imagem não reconhecida. A confiança da previsão está muito baixa.")  # Exibe a mensagem de imagem não reconhecida.
            else:
                st.error("O modelo não foi carregado corretamente. Verifique o arquivo VGG16.h5.")  # Exibe um erro se o modelo não foi carregado.
        else:
            st.error("Erro ao pré-processar a imagem. Verifique o formato da imagem.") # Exibe um erro se a imagem não pode ser pre processada

    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")  # Exibe uma mensagem de erro se ocorrer um erro durante o processamento.
content_copy
download
Use code with caution.
Python

Resumo do Código:

Este script Streamlit cria um dashboard interativo para diagnosticar doenças em folhas de cana-de-açúcar, permitindo que o usuário carregue uma imagem, a qual é processada e classificada por um modelo CNN (VGG16.h5). O dashboard exibe a imagem, o nome da doença ou a indicação de que a folha está saudável, a probabilidade da previsão e informações adicionais sobre a doença. Um histórico das previsões é salvo em um arquivo CSV.

Aprimoramentos:

Tratamento de Erros Abrangente: Adicionado tratamento de erros em todas as funções principais (carregamento do modelo, pré-processamento da imagem, previsão e salvamento do histórico) para evitar falhas inesperadas.

Validação do Pré-Processamento: Verificação se a imagem foi corretamente pré-processada antes de prosseguir com a previsão.

Mensagem de "Imagem Não Reconhecida": Implementada a lógica para exibir uma mensagem amigável ao usuário quando a confiança da previsão estiver abaixo de um limiar, indicando que o modelo não conseguiu identificar a imagem.

Código Comentado: Cada linha de código foi comentada para facilitar a compreensão e manutenção.

Para usar, certifique-se de ter as bibliotecas instaladas e que o arquivo VGG16.h5 esteja na mesma pasta do script. Execute com streamlit run app.py.