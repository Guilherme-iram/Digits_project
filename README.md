# Classificação de Digitos com Modelos Lineares 

![Alt text](./imgs/image.png)

## Algoritmos e estratégias de classificação utilizadas
- Regressão Linear;
- Regressão Logistica;
- Perceptron 2D;
- Um contra todos.

## Dataset
- É uma adaptação do clássico MNIST Dataset, no qual cada classe é linearmente separável 2 a 2.
<br></br>
![Alt text](./imgs/image-1.png)

## Pré-processamento de Features
- Foi aplicada uma redução de dimensionalidade nos registros do dataset;
- Os 782 pixels foram transformados em 2 features: intensidade e simetria.
<br></br>
![Alt text](./imgs/image-2.png)
![Alt text](./imgs/image-3.png)

## Classifcando o digito 1 e 5

- O primeiro teste dos modelos foi classificar os registros de label 1 e 5

![Alt text](./imgs/image-8.png)
---
- Dados de treino utilizados:
![Alt text](./imgs/image-4.png)

### Reta da _Regressão Linear_
![Alt text](./imgs/image-5.png)
### Reta do _PLA_
![Alt text](./imgs/image-6.png)
### Reta da _Regressão Logística_
![Alt text](./imgs/image-7.png)

## Resultado final

- Para efetuar a classificação de todos os digitos (4 labels no total), foi utilizada a estratégia de classificação **"Um contra todos"**.

![Alt text](./imgs/image-9.png)
- E o algoritmo seguido foi o seguinte: 
```
para os dígitos 𝑖 ∈ [0,1,4]
    se 𝑓𝑖(𝑥) = +1
        classifique como dígito i
    senão
        se i == 4
            classifique como dígito 5
```
- Os resultados obtidos são os que se seguem para o uso de cada modelo:

![Alt text](./imgs/image-10.png)
---

