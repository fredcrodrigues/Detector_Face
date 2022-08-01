## Projeto Detector de Mascara - Aprendizado Profundo

Este projeto é baseado em *Machine Learning* e na linguagem de programação python. A proposta é realizar a detecção de pessoas **USANDO MÁSCARA** em ambientes abertos ou frechados. Esse projeto está dentro do contexto pandemico ocorrido entre o ano de 2019-2022.
A base de dados selecionada pode ser adquirida neste [Link](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection). Dessa forma é gerado um modelo de predição para detectar pessoas usando mascaras ou não.


## Imagens da base de dados
[!Screenshot](/img/img1.png)

## Execuçaõp do Projeto

```bash
    tensorflow = 2.5.0
    opencv-python = 4.6.0
    python = 3.8.8
    numpy = 1.21
```

A arquitetura para treino e geração do modelo é a Mobilinet, metodo de otimização Adam e treinamento em 100 épocas.


## Resultados do Detector
** Teste em imagens da base** (Essas imagens não são sadas para treinamento)
[!Screenshot /img/saida.png]

** Teste em imagens de video capturada pela webcam em laboratório** 
[!Screenshot /img/saida2.png]
