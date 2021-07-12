---
layout: post
title: "Florestas Aleatórias"
date: 2020-09-09 23:07:41 
lang: R 
category: Métodos de Treino Baseados em Árvores
description: "Criação: Maqueise Pinheiro e Thaís Machado. Orientação: Douglas Rodrigues (UFF) e Karina Yaginuma (UFF). Colaboradores: Gabriel Miranda e Thiago Augusto."
---

### Florestas Aleatórias

As árvores de decisão possuem uma estrutura de fácil compreensão, o que faz com que ela seja bastante utilizada devido a sua boa aparência e interpretação intuitiva. Mas elas possuem uma limitação, o [sobreajuste](#link4). Sendo assim, elas não são muito eficientes com novas amostras. O que fazer então?

As **Florestas Aleatórias** (*Random Forests*) se utilizam de várias árvores de decisão, combinando a simplicidade das árvores com a flexibilidade de um método sem sobreajuste, aumentando assim a precisão do preditor.

Vamos construir uma floresta aleatória usando a base de dados `balloons`.


```r
balloons = readr::read_csv("balloons.csv")
balloons$Inflated = as.factor(balloons$Inflated)
str(balloons)
```

```
## tibble [20 x 5] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
##  $ Color   : chr [1:20] "YELLOW" "YELLOW" "YELLOW" "YELLOW" ...
##  $ Size    : chr [1:20] "SMALL" "SMALL" "SMALL" "SMALL" ...
##  $ Act     : chr [1:20] "STRETCH" "STRETCH" "STRETCH" "DIP" ...
##  $ Age     : chr [1:20] "ADULT" "ADULT" "CHILD" "ADULT" ...
##  $ Inflated: Factor w/ 2 levels "FALSE","TRUE": 2 2 1 1 1 2 2 1 1 1 ...
##  - attr(*, "spec")=
##   .. cols(
##   ..   Color = col_character(),
##   ..   Size = col_character(),
##   ..   Act = col_character(),
##   ..   Age = col_character(),
##   ..   Inflated = col_logical()
##   .. )
```

Com base na cor do balão, no tamanho dele, se ele é elástico ou não e se quem o está enchendo é uma criança ou um adulto, queremos predizer se o balão vai encher ou não. Portanto, nossa variável de interesse é `Inflated` e queremos construir um classificador.

A primeira coisa que precisamos fazer é criar uma nova amostra do mesmo tamanho da original utilizando [bootstrap](#link5).


```r
set.seed(33)
boot1 = caret::createResample(y=balloons$Inflated, times=1, list=F)
NovaAmostra1 = balloons[boot1,]
Out_of_bag = balloons[-boot1,]
```

Todas as observações que não forem sorteadas vão entrar no "*Out-of-Bag*". Temos 4 variáveis fora a de interesse, vamos sortear 2 variáveis para construir o primeiro nó da nossa árvore.


```r
set.seed(413)
sample(1:4, 2)
```

```
## [1] 4 3
```

Vamos calcular o índice Gini para essas duas variáveis.


```r
#calculando o indice gini para a variável tamanho
table(NovaAmostra1$Size, NovaAmostra1$Inflated)
```

```
##        
##         FALSE TRUE
##   LARGE     9    4
##   SMALL     4    3
```

```r
(gini.size = (1-(7/14)^2-(7/14)^2)*(14/20) + (1-(4/6)^2-(2/6)^2)*(6/20))
```

```
## [1] 0.4833333
```

```r
#calculando o indice gini para a variável idade
table(NovaAmostra1$Age, NovaAmostra1$Inflated)
```

```
##        
##         FALSE TRUE
##   ADULT     4    7
##   CHILD     9    0
```

```r
(gini.age = (1-(5/14)^2-(9/14)^2)*(14/20) + (1-(6/6)^2-(0/6)^2)*(6/20))
```

```
## [1] 0.3214286
```

A variável idade tem um grau de impureza menor, então ela será a raiz da árvore.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/floresta1.jpeg)

Agora das variáveis que ainda não foram usadas, sorteamos mais duas para continuar a árvore.


```r
set.seed(443)
sample(1:3, 2)
```

```
## [1] 3 2
```


```r
library(dplyr)
NovaAmostra1 = filter(NovaAmostra1, Age=="ADULT") 
#calculando o indice gini para a variável tamanho
table(NovaAmostra1$Size, NovaAmostra1$Inflated)
```

```
##        
##         FALSE TRUE
##   LARGE     3    4
##   SMALL     1    3
```

```r
(gini.size = (1-(4/11)^2-(7/11)^2)*(11/14) + (1-(1/3)^2-(2/3)^2)*(3/14))
```

```
## [1] 0.4588745
```

```r
#calculando o indice gini para a variável act
table(NovaAmostra1$Act, NovaAmostra1$Inflated)
```

```
##          
##           FALSE TRUE
##   DIP         4    0
##   STRETCH     0    7
```

```r
(gini.act = (1-(5/5)^2-(0/5)^2)*(5/14) + (1-(0/9)^2-(9/9)^2)*(9/14))
```

```
## [1] 0
```

Como a variável *act* tem o menor grau de impureza, ela será o próximo nó.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/floresta2.jpeg)

Assim, temos nossa primeira árvore de decisão.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/floresta3.jpeg)

> A floresta aleatória pode ser utilizada tanto em classificadores como em regressores. A diferença é que em regressores, utilizamos [árvores de regressão](#link11) no lugar de [árvores de classificação](#link12). 

Em seguida vamos construir várias árvores da mesma maneira que a anterior. Para nosso exemplo vamos construir apenas 4 árvores, mas em geral vamos fazer bem mais que isso.

As 4 árvores construídas ficam da seguinte forma:

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/floresta4.jpeg)

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/floresta5.jpeg)

Para classificar uma nova amostra, devemos passar ela por todas as árvores construidas e rotular a amostra pela categoria resultada mais vezes.

> O método de usar bootstrap para criar novas amostras e votos para a tomada de decisão é chamado de ***Bagging*** (***B**ootstrap+**agg**regate*).

As observações de cada amostra que não entraram na construção de cada árvore estão contidas `Out_of_bag`. Essas observações servirão para avaliar nosso preditor.


```r
Out_of_bag = balloons[c(2,4,12,13,15,18,20,
                        1,2,3,5,10,
                        2,4,12,13,15,18,20,
                        2,3,11,13,14,16,19),]
knitr::kable(Out_of_bag)
```



|Color  |Size  |Act     |Age   |Inflated |
|:------|:-----|:-------|:-----|:--------|
|YELLOW |SMALL |STRETCH |ADULT |TRUE     |
|YELLOW |SMALL |DIP     |ADULT |FALSE    |
|PURPLE |SMALL |STRETCH |ADULT |TRUE     |
|PURPLE |SMALL |STRETCH |CHILD |FALSE    |
|PURPLE |SMALL |DIP     |CHILD |FALSE    |
|PURPLE |LARGE |STRETCH |CHILD |FALSE    |
|PURPLE |LARGE |DIP     |CHILD |FALSE    |
|YELLOW |SMALL |STRETCH |ADULT |TRUE     |
|YELLOW |SMALL |STRETCH |ADULT |TRUE     |
|YELLOW |SMALL |STRETCH |CHILD |FALSE    |
|YELLOW |SMALL |DIP     |CHILD |FALSE    |
|YELLOW |LARGE |DIP     |CHILD |FALSE    |
|YELLOW |SMALL |STRETCH |ADULT |TRUE     |
|YELLOW |SMALL |DIP     |ADULT |FALSE    |
|PURPLE |SMALL |STRETCH |ADULT |TRUE     |
|PURPLE |SMALL |STRETCH |CHILD |FALSE    |
|PURPLE |SMALL |DIP     |CHILD |FALSE    |
|PURPLE |LARGE |STRETCH |CHILD |FALSE    |
|PURPLE |LARGE |DIP     |CHILD |FALSE    |
|YELLOW |SMALL |STRETCH |ADULT |TRUE     |
|YELLOW |SMALL |STRETCH |CHILD |FALSE    |
|PURPLE |SMALL |STRETCH |ADULT |TRUE     |
|PURPLE |SMALL |STRETCH |CHILD |FALSE    |
|PURPLE |SMALL |DIP     |ADULT |FALSE    |
|PURPLE |LARGE |STRETCH |ADULT |TRUE     |
|PURPLE |LARGE |DIP     |ADULT |FALSE    |

Para avaliar, é preciso passar cada uma das observações do `Out_of_bag` por todas as árvores e a predição será feita por votos também. Ao fazer isso, observamos uma [precisão](#link6) de 86%.

> A proporção de amostras do Out-of-bag que foram incorretamente classificadas é chamada *Out-of-bag-error*

Agora que sabemos avaliar o modelo, podemos comparar florestas aleatórias construídas com 2 variáveis com as construídas com 3 e outras diferentes configurações. Tipicamente, começamos usando o quadrado do número de variáveis da base e tentamos algumas quantidades abaixo e acima.

#### Construindo uma floresta com o `randomForest()`

O pacote `randomForest`[@randomforest] possui as ferramentas adequadas para a criação de uma floresta aleatória. Vamos construir uma floresta com 20 árvores utilizando a base `balloons`.

> É importante observar se as váriaveis categóricas estão na classe de fatores.


```r
balloons = readr::read_csv("balloons.csv")
# Tratando todas as variáveis: 
balloons = dplyr::mutate_if(balloons, is.character, as.factor)
balloons$Inflated = as.factor(balloons$Inflated)
# Construindo uma floresta com 20 árvores:
library(randomForest)
set.seed(23)
modelo = randomForest(Inflated ~ ., data=balloons, ntree=20)
```

Agora, vamos avaliar a precisão do modelo.


```r
modelo
```

```
## 
## Call:
##  randomForest(formula = Inflated ~ ., data = balloons, ntree = 20) 
##                Type of random forest: classification
##                      Number of trees: 20
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 5%
## Confusion matrix:
##       FALSE TRUE class.error
## FALSE    11    1  0.08333333
## TRUE      0    8  0.00000000
```

> Note que foram construídas 20 árvores utilizando 2 variáveis a cada vez. Essa quantidade de variáveis pode ser alterada usando o argumento `mtry` dentro do randomForest.

Podemos ver que a precisão do nosso modelo é de 19/20, ou seja, 95%. Qual seria a precisão se fosse feito apenas uma árvore?


```r
balloons = readr::read_csv("balloons.csv")

# Tratando todas as variáveis: 
balloons = dplyr::mutate_if(balloons, is.character, as.factor)
balloons$Inflated = as.factor(balloons$Inflated)

# Separando amostras teste/treino:
set.seed(45)
inTrain = caret::createDataPartition(balloons$Inflated,p=0.5,list=F)
treino = balloons[inTrain,]
teste = balloons[-inTrain,]

# Treinando o modelo:
controle = rpart::rpart.control(minsplit=0, cp = 0, maxdepth = 1)
set.seed(342)
modelo = rpart::rpart(Inflated~., data=treino, control = controle)

# Aplicando o modelo no teste:
predicao = predict(modelo,teste, type="vector")
predicao = factor(predicao, labels = c(F, T))

# Avaliando o erro:
confusionMatrix(predicao, teste$Inflated)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction FALSE TRUE
##      FALSE     3    0
##      TRUE      3    4
##                                           
##                Accuracy : 0.7             
##                  95% CI : (0.3475, 0.9333)
##     No Information Rate : 0.6             
##     P-Value [Acc > NIR] : 0.3823          
##                                           
##                   Kappa : 0.4444          
##                                           
##  Mcnemar's Test P-Value : 0.2482          
##                                           
##             Sensitivity : 0.5000          
##             Specificity : 1.0000          
##          Pos Pred Value : 1.0000          
##          Neg Pred Value : 0.5714          
##              Prevalence : 0.6000          
##          Detection Rate : 0.3000          
##    Detection Prevalence : 0.3000          
##       Balanced Accuracy : 0.7500          
##                                           
##        'Positive' Class : FALSE           
## 
```

Note que nessa árvore o modelo teve uma precisão de 80%. Bem menor do que o modelo com florestas.

Agora observe que anteriormente construímos uma floresta com 20 árvores. O que acontece com o erro do modelo conforme acrescentamos mais árvores?

Vamos avaliar o comportamento do erro conforme acrescentamos mais árvores à floresta. Para isso utilizaremos a base de dados "spam" para melhor vizualização.


```r
library(kernlab)
data("spam")
# Construindo uma floresta com 20 árvores:
library(randomForest)
set.seed(23)
modelo = randomForest(type ~ ., data=spam, ntree=20)

# Observando o comportamento do erro em 20 árvores:
erro_OOB <- data.frame(
  Arvores = rep(1:nrow(modelo$err.rate), times=2),
  Type = rep(c("spam", "nonspam"), each=nrow(modelo$err.rate)),
  Erro = c(modelo$err.rate[,"spam"], modelo$err.rate[,"nonspam"]))

ggplot(data=erro_OOB, aes(x=Arvores, y=Erro)) +
  geom_line(aes(color=Type),size=1.1) +
  scale_colour_discrete(name = "Tipo",
                         breaks = c("nonspam", "spam"),
                         labels = c("Não Spam", "Spam"))
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-149-1.png)


```r
# Construindo uma floresta com 50 árvores:
set.seed(23)
modelo = randomForest(type ~ ., data=spam, ntree=50)

# Observando o comportamento do erro em 50 árvores:
erro_OOB <- data.frame(
  Arvores=rep(1:nrow(modelo$err.rate), times=2),
  Type=rep(c("spam", "nonspam"), each=nrow(modelo$err.rate)),
  Erro=c(modelo$err.rate[,"spam"],
          modelo$err.rate[,"nonspam"]))

ggplot(data=erro_OOB, aes(x=Arvores, y=Erro)) +
  geom_line(aes(color=Type),size=1.1)+
  scale_colour_discrete(name = "Tipo",
                         breaks = c("nonspam", "spam"),
                         labels = c("Não Spam", "Spam"))
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-150-1.png)


```r
# Construindo uma floresta com 100 árvores:
set.seed(23)
modelo = randomForest(type ~ ., data=spam, ntree=100)

# Observando o comportamento do erro em 100 árvores:
erro_OOB <- data.frame(
  Arvores=rep(1:nrow(modelo$err.rate), times=2),
  Type=rep(c("spam", "nonspam"), each=nrow(modelo$err.rate)),
  Erro=c(modelo$err.rate[,"spam"],
          modelo$err.rate[,"nonspam"]))

ggplot(data=erro_OOB, aes(x=Arvores, y=Erro)) +
  geom_line(aes(color=Type),size=1.1)+
  scale_colour_discrete(name = "Tipo",
                         breaks = c("nonspam", "spam"),
                         labels = c("Não Spam", "Spam"))
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-151-1.png)


```r
# Construindo uma floresta com 1000 árvores:
set.seed(23)
modelo = randomForest(type ~ ., data=spam, ntree=1000)

# Observando o comportamento do erro em 1000 árvores:
erro_OOB <- data.frame(
  Arvores=rep(1:nrow(modelo$err.rate), times=2),
  Type=rep(c("spam", "nonspam"), each=nrow(modelo$err.rate)),
  Erro=c(modelo$err.rate[,"spam"],
          modelo$err.rate[,"nonspam"]))

ggplot(data=erro_OOB, aes(x=Arvores, y=Erro)) +
  geom_line(aes(color=Type),size=1.1)+
  scale_colour_discrete(name = "Tipo",
                         breaks = c("nonspam", "spam"),
                         labels = c("Não Spam", "Spam"))
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-152-1.png)

Repare que após uma certa quantidade de árvores o erro se estabiliza. Sendo assim, não é necessário utilizar grandes quantidades de árvores em todos os casos. É preciso verificar até onde existe ganho.

#### Construindo uma floresta com o `train()`

Também é possivel fazer florestas aleatórias usando a função `train` do pacote `caret`. Para isso, é necessário alterar o método de reamostragem para *out of bag* e o método para "rf" (*random forest*). Vamos utilizar a base `wine`, onde construiremos um regressor para predizer a variável `alcohol`.


```r
# Alterando o método de reamostragem:
controle = trainControl(method="oob")

# Carregando a base:
library(readr)
wine = read_csv2("winequality-red.csv")

# Construindo o modelo com 50 árvores:
set.seed(534)
modelo = caret::train(alcohol~., data=wine, method="rf", ntree=50, trControl=controle)
modelo
```

```
## Random Forest 
## 
## 1599 samples
##   11 predictor
## 
## No pre-processing
## Resampling results across tuning parameters:
## 
##   mtry  RMSE       Rsquared 
##    2    0.5546635  0.7289263
##    6    0.5028345  0.7772189
##   11    0.5034648  0.7766601
## 
## RMSE was used to select the optimal model using the smallest value.
## The final value used for the model was mtry = 6.
```

Note o valor "`mtry`" no modelo. Ele indica a quantidade de váriaveis da base que foram utilizadas para treinar o modelo. Repare que ele calcula o RMSE e o $R^2$ para diferentes quantidades de variáveis usadas e utiliza no final a quantidade que possuir menor RMSE, no caso mtry=11. Caso queira fixar o número de variáveis usadas, basta usar o seguinte comando.


```r
tng = expand.grid(.mtry=7)
modelo = caret::train(alcohol~., data=wine, method="rf", ntree=50, trControl=controle, tuneGrid=tng)
modelo
```

```
## Random Forest 
## 
## 1599 samples
##   11 predictor
## 
## No pre-processing
## Resampling results:
## 
##   RMSE       Rsquared 
##   0.5127032  0.7683885
## 
## Tuning parameter 'mtry' was held constant at a value of 7
```

---