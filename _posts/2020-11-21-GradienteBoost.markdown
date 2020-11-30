---
layout: post
title: "Gradiente Boosting"
date: 2020-09-09 23:07:41 
lang: R 
category: Métodos de Treino Baseados em Árvores
description: "Criação: Maqueise Pinheiro e Thaís Machado. Orientação: Douglas Rodrigues (UFF) e Karina Yaginuma (UFF). Colaboradores: Gabriel Miranda e Thiago Augusto."
---

### *Gradiente Boosting*

#### em Regressão

De acordo com Jerome Friedman, o criador do Gradiente Boosting, evidências empíricas mostram que dar pequenos passos ou ir gradativamente na direção correta resulta em melhores predições na amostra teste, ou seja, menor variância.

Para entendermos como funciona o Gradiente Boosting, considere a seguinte base de dados.


```
##   Altura      Cor Sexo Peso
## 1    1.6     Azul    M   88
## 2    1.6    Verde    F   76
## 3    1.5     Azul    F   56
## 4    1.8 Vermelho    M   75
## 5    1.5    Verde    M   77
## 6    1.4     Azul    F   57
```

A primeira coisa a fazer é definir um número máximo de folhas de cada árvore. Para nosso exemplo, vamos definir 4 folhas, mas em geral, é definido uma quantidade de 8 a 32 folhas. Feito isso, tiramos uma média dos pesos dos indivíduos e essa será nossa primeira árvore, uma árvore só com a raiz. 


![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/tocoGB.png)



Agora, calculamos o **Pseudo-Resíduo**, o erro de previsão de cada indivíduo, da forma $$Pseudo\ Resíduo = valor\ real - valor\ predito$$ para cada observação. Então, por exemplo, o pseudo-resíduo da primeira observação vai ficar $88-71.5 = 16.5$ .


```
##   Altura      Cor Sexo Peso Ps..Res..1
## 1    1.6     Azul    M   88       16.5
## 2    1.6    Verde    F   76        4.5
## 3    1.5     Azul    F   56      -15.5
## 4    1.8 Vermelho    M   75        3.5
## 5    1.5    Verde    M   77        5.5
## 6    1.4     Azul    F   57      -14.5
```

> O termo **pseudo-resíduo** é baseado em Regressão Linear, onde o resíduo é a diferença entre os valores observados e estimados. O termo "pseudo" serve para lembrar que estamos fazendo **Gradiente Boosting** e não Regressão Linear.

O próximo passo é, utilizando as variáveis explicativas (Altura, Cor e Sexo), construir uma árvore de decisão respeitando o máximo de folhas definido anteriormente. Mas ela deve **predizer o pseudo-resíduo** e não o Peso.


![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/GB1.png)



> Note que temos mais observações do que folhas, sendo assim, podemos ter mais que um resultado em cada uma. Nesse caso, substituímos os valores pela média das folhas.


![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/GB2.png)



Agora somamos o resultado das duas árvores para classificar na primeira observação, por exemplo, a predição seria $71.5+16.5=88$. Acertamos exatamente o valor real. Isso é bom? Não. Já vimos como não é bom ter um modelo muito ajustado. Temos pouco viés, mas provavelmente alta variância.

O Gradiente Boosting lida com esse problema usando uma **taxa de aprendizado** para reescalar a contribuição da nova árvore. A taxa de aprendizado é um número entre 0 e 1 e deve ser multiplicado ao valor da segunda árvore em diante. Para esse exemplo, vamos adotar uma taxa de 0.1, assim a predição da primeira observação seria $71.5+(0.1*16.5)=73.15$. A predição não ficou tão boa, mas é um pouco melhor do que o resultado de apenas uma árvore.

Feito isso, recalculamos os valores do pseudo-resíduo.


```
##   Altura      Cor Sexo Peso Ps..Res..1 Ps..Res..2
## 1    1.6     Azul    M   88       16.5      14.85
## 2    1.6    Verde    F   76        4.5       4.05
## 3    1.5     Azul    F   56      -15.5     -14.00
## 4    1.8 Vermelho    M   75        3.5       3.05
## 5    1.5    Verde    M   77        5.5       5.05
## 6    1.4     Azul    F   57      -14.5     -13.00
```

Repare que o valor do segundo pseudo-resíduo diminuiu em módulo em relação ao primeiro, ou seja, nos aproximamos mais do valor correto do que da primeira vez.

Agora, utilizando novamente as variáveis explicativas, construímos outra árvore agora para predizer o segundo pseudo-resíduo. 


![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/GB3.png)



> Note que a estrutura da segunda árvore construída ficou semelhante a primeira. Isso não acontece sempre, mas pode acontecer.

Agora, a classificação da primeira observação ficaria $71.5+(0.1*16.5)+(0.1*14.85)=74.635$  um pouco mais perto do verdadeiro valor. Repetimos esse procedimento quantas vezes se queira ou até não ter redução significante dos valores do pseudo-resíduo. Dessa forma temos uma sequência
de árvores que caminham em direção ao valor correto em passos pequenos.

> É importante notar que todas as árvores devem possuir a mesma taxa de aprendizado.

##### Construindo um regressor com o pacote `gbm`

Para nosso exemplo, vamos utilizar a base de dados `Wage` do pacote `ISLR`. Para isso, vamos precisar limpar os dados removendo [variáveis de variância zero](#link7).


```r
# lendo a base de dados
library(ISLR)
data("Wage")
# removendo variaveis de variancia zero
(vvz = nearZeroVar(Wage,saveMetrics = F))
```

```
## [1] 6
```

```r
Wage = Wage[,-vvz]
```

Dividimos a base nos conjuntos de treino e teste.


```r
set.seed(100)
noTreino = createDataPartition(Wage$wage,p=0.7,list=F)
treino = Wage[noTreino,]
teste = Wage[-noTreino,]
```

Agora vamos aplicar o gradiente boosting utilizando o pacote `gbm`[@gbm].


```r
library(gbm)
set.seed(100)
(modelo = gbm(wage~.,data=treino, distribution="gaussian",
              n.trees =300,interaction.depth = 20))
```

```
## gbm(formula = wage ~ ., distribution = "gaussian", data = treino, 
##     n.trees = 300, interaction.depth = 20)
## A gradient boosted model with gaussian loss function.
## 300 iterations were performed.
## There were 9 predictors of which 9 had non-zero influence.
```

Os principais argumentos da função `gbm()` são:

+ **distribution**: *gaussian* se for regressão, *multinomial* se for um classificação, *bernoulli* se for classificação 0-1.
+ **n.trees**: número de árvores da floresta.
+ **interaction.depth**: profundidade máxima das árvores.

Vamos aplicar o modelo na amostra teste, e avaliar o resultado.


```r
predicao = predict(modelo, teste, n.trees=300)
# avaliando
postResample(predicao,teste$wage)
```

```
##      RMSE  Rsquared       MAE 
## 0.7297289 0.9996961 0.1742759
```

```r
# ou
caret::defaultSummary(data.frame(obs = teste$wage, pred = predicao))
```

```
##      RMSE  Rsquared       MAE 
## 0.7297289 0.9996961 0.1742759
```

Note que utilizamos 300 árvores. Mas pode ser que não seja necessário essa quantidade de árvores pra alcançar esses valores de $R^2$, RMSE e MAE. Para saber a quantidade ideal de árvores, isto é, quando erro se estabiliza, podemos utilizar a função `gbm.perf()`.


```r
gbm.perf(modelo)
```

![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/unnamed-chunk-177-1.png)

```
## [1] 48
## attr(,"smoother")
## Call:
## loess(formula = object$oobag.improve ~ x, enp.target = min(max(4, 
##     length(x)/10), 50))
## 
## Number of Observations: 300 
## Equivalent Number of Parameters: 24.11 
## Residual Standard Error: 5.161
```

Sendo assim, com apenas 50 árvores teríamos chegado a um resultado razoável.


```r
predicao2 = predict(modelo, teste, n.trees=50)
postResample(predicao2,teste$wage)
```

```
##      RMSE  Rsquared       MAE 
## 1.0759904 0.9993670 0.2823224
```

Podemos ver que o RMSE e o MAE aumentaram um pouco e o $R^{2}$ foi praticamente o mesmo. Mas como tivemos um custo computacional muito menor, podemos concluir que esse modelo com 50 árvores acaba sendo melhor do que o com 300.

#### em Classificação

Considere a seguinte base de dados


```
## # A tibble: 6 x 4
##   `Gosta de Pipoca` Idade `Cor Favorita` `Troll 2`
##   <chr>             <dbl> <chr>          <chr>    
## 1 Sim                  12 Azul           Ama      
## 2 Sim                  87 Verde          Ama      
## 3 Nao                  44 Azul           Odeia    
## 4 Sim                  19 Vermelho       Odeia    
## 5 Nao                  32 Verde          Ama      
## 6 Nao                  14 Azul           Ama
```

Queremos predizer se uma pessoa ama o filme Troll 2 baseado em seu gosto por pipoca, idade e cor favorita. Assim como em regressão, começamos o método de Gradiente Boosting usando uma árvore raiz que represente nossa predição inicial para cada observação. Em regressão usamos a média das observações, em classificação vamos usar o log(chances). Olhando na base de dados, podemos dizer que as chances de alguém amar Troll 2 é $$chances= \frac{Quantidade\ de\ indivíduos\ que\ amaram}{Quantidade\ de\ indivíduos\ que\ odiaram} = \frac{4}{2}$$ portanto, o $log(chances)=log(\frac{4}{2}) = 0.6932$ e é isso que colocaremos na folha inicial.

![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/tocoGBC.png)


O jeito mais fácil de usar o log(chances) para classificar é convertendo em probabilidade, e fazemos isso usando a seguinte função: $$ Probabilidade = \frac{e^{log(chances)}}{1+e^{log(chances)}}$$
Sendo assim, A $Probabilidade\ de\ alguém\ amar\ Troll2 = \frac{e^{log(\frac{4}{2})}}{1+e^{log(\frac{4}{2})}} = \frac{2}{3}=0.6667$. 

![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/tocoGBC2.png)


> É importante notar que o log(chances) e a probabilidade só ficaram iguais por causa da aproximação.

Vamos criar o seguinte classificador:

- Probabilidade acima de 0,5: classificamos que ama Troll 2;
- Probabilidade menor ou igual a 0,5: classificamos que odeia Troll 2.

Como a probabilidade ficou maior que 0,5 classificamos todos no treino como indivíduos que amam Troll 2.

> Embora 0,5 seja um limite usual para tomada de decisão baseada em probabilidade, poderiamos tranquilamente usar um valor diferente.

Mas a classificação não ficou muito boa já que 2 indivíduos foram classificados erroneamente. Podemos mensurar quão ruim foi a predição calculando o $pseudo\ resíduo = lod(odds)\ observado - predito$. Para essa conta, perceba que se um indivíduo ama Troll 2, então a probabilidade dele amar Troll 2 é 1. Semelhantemente, se ele odeia, a probabilidade dele amar é 0. Assim, calculamos os pseudo-resíduos.


```
## # A tibble: 6 x 5
##   `Gosta de Pipoca` Idade `Cor Favorita` `Troll 2` `Ps. Res. 1`
##   <chr>             <dbl> <chr>          <chr>            <dbl>
## 1 Sim                  12 Azul           Ama                0.3
## 2 Sim                  87 Verde          Ama                0.3
## 3 Nao                  44 Azul           Odeia             -0.7
## 4 Sim                  19 Vermelho       Odeia             -0.7
## 5 Nao                  32 Verde          Ama                0.3
## 6 Nao                  14 Azul           Ama                0.3
```

Agora construímos uma árvore utilizando as variáveis explicativas para predizer o pseudo-resíduo. Assim como o Gradiente Boosting para regressão, temos que definir um número máximo de folhas em cada árvore. Aqui vamos limitar a 3 folhas, mas na prática geralmente é um número entre 8 e 32.

![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/tocoGBC3.png)


Em regressão, os valores das folhas representavam os resíduos. Mas em classificação isso é mais complexo. Isso porque a predição está em log(chances) e as folhas são provenientes de probabilidade. Portanto não podemos apenas tirar a média para uma nova predição sem alguma transformação. A transformação mais comum por folha é $$\frac{\sum resíduos}{\sum[prob.\ anterior * (1-prob.\ anterior)]}$$

Assim, da esquerda pra direita, para primeira folha temos $$\frac{\sum resíduos}{\sum[prob.\ anterior * (1-prob.\ anterior)]}=\frac{-0.7}{0.7*(1-0.7)}=-3.3333,$$ para a segunda $$\frac{\sum resíduos}{\sum[prob.\ anterior * (1-prob.\ anterior)]}=\frac{0.3+(-0.7)}{(0.7*(1-0.7))+(0.7*(1-0.7))}=-0.9524$$ e para a última $$\frac{\sum resíduos}{\sum[prob.\ anterior * (1-prob.\ anterior)]}=$$ $$\frac{0.3+0.3+0.3}{(0.7*(1-0.7))+(0.7*(1-0.7))+(0.7*(1-0.7))}=\frac{3*0.3}{3*(0.7*(1-0.7))}=1.4286$$

> Por enquanto, a probabilidade anterior é a mesma para todos, mas a partir da próxima árvore isso muda.

![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/tocoGBC4.png)


Agora que todas as folhas foram alteradas, podemos somar os resultados escalados pela taxa de aprendizado. Nesse exemplo, vamos usar uma taxa alta, 0.8. Mas geralmente se usa 0.1. E então calculamos o novo $log(chances)=log(chances)\ anterior + taxa\ de\ aprendizado * log(chances)\ obtido\ na\ árvore$. Para primeira observação, por exemplo, fica $log(chances)=0.7+(0.8*1.4)=1.82$  e então convertemos em probabilidade $\frac{e^{1.82}}{1+e^{1.82}} = 0.8606$. Então, note que fizemos progresso, já que o indivíduo em questão ama Troll 2. Antes ele foi classificado corretamente mas com probabilidade 0.7, agora ele foi classificado corretamente mas com probabilidade 0.9.


```
## # A tibble: 6 x 5
##   `Gosta de Pipoca` Idade `Cor Favorita` `Troll 2` `Prob. Predita`
##   <chr>             <dbl> <chr>          <chr>               <dbl>
## 1 Sim                  12 Azul           Ama                   0.9
## 2 Sim                  87 Verde          Ama                   0.5
## 3 Nao                  44 Azul           Odeia                 0.5
## 4 Sim                  19 Vermelho       Odeia                 0.1
## 5 Nao                  32 Verde          Ama                   0.9
## 6 Nao                  14 Azul           Ama                   0.9
```

> Pode ser que a previsão fique pior, como no caso do segundo indivíduo. E essa é a razão de construírmos várias árvores e não só uma.

Calculamos os novos pseudo-resíduos que agora serão diferentes para cada observação.


```
## # A tibble: 6 x 5
##   `Gosta de Pipoca` Idade `Cor Favorita` `Troll 2` `Ps. Res. 2`
##   <chr>             <dbl> <chr>          <chr>            <dbl>
## 1 Sim                  12 Azul           Ama                0.1
## 2 Sim                  87 Verde          Ama                0.5
## 3 Nao                  44 Azul           Odeia             -0.5
## 4 Sim                  19 Vermelho       Odeia             -0.1
## 5 Nao                  32 Verde          Ama                0.1
## 6 Nao                  14 Azul           Ama                0.1
```

Construímos uma segunda árvore agora para prever os novos pseudo-resíduos e fazemos a transformação para log(chances) para cada folha.

![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/tocoGBC5.png)


Combinamos com as árvores anteriores para obter um valor de saída e transformamos em Probabilidade para classificar. Por exemplo, a primeira observação ficaria $log(chances)=0.7+(0.8*1.4)+(0.8*0.6)=2.3$ e então convertemos em probabilidade $\frac{e^{1.82}}{1+e^{1.82}} = 0.9089$. Dessa forma, continuamos construíndo quantas árvores forem necessárias.

#### Construindo um classificador com o pacote `gbm`

O gradiente boosting para classificação no R é semelhante ao para regressão, atentando para o argumento `distribution`, que deve ser igual a "bernoulli" se a variável de interesse tiver apenas duas respostas possíveis, como no caso da base Troll 2 (nesse caso, a variavel de interesse deve ser uma indicadora, ou seja, assumir valores 0's e 1's) ou "multinomial" se a variável tiver mais de duas respostas possíveis. Por exemplo, considere a base `Vehicle` do pacote `mlbench`[@mlbench]. Nela, estamos interessados em classificar a variável `Class`, que pode ser *bus, opel, saab* ou *van*.


```r
# lendo a base
library(mlbench)
data(Vehicle)
# dividindo em treino e teste
library(caret)
set.seed(100)
noTreino = createDataPartition(Vehicle$Class,p=0.7,list=F)
treino = Vehicle[noTreino,]
teste = Vehicle[-noTreino,]
# treinando o modelo
library(gbm)
set.seed(100)
modelo = gbm(Class~.,data=treino,distribution="multinomial",
              n.trees = 100,interaction.depth = 8)
```

Quando aplicamos o `predict()`, o que recebemos de retorno são um conjunto de probabilidades (ou o log(chances)), e não a classificação final. Cabe ao pesquisador definir a regra de classificação final.


```r
predicao = predict(modelo, teste, n.trees = 100, type = 'response')
```

> O argumento ´type´ retorna por default o log(chances), se definimos como "response" ele retorna a probabilidade.


```r
# Criando a regra de classificacao
k = dim(teste)[1]
classe = c()
for (i in 1:k){
  classe[i] = names(which.max(predicao[i,1:4,1])) 
}
head(classe)
```

```
## [1] "van"  "van"  "opel" "van"  "bus"  "saab"
```

```r
# verificando quantidade de arvores necessarias
gbm.perf(modelo)
```

![tela_0]({{ site.url }}/assets/python/courses/machine_learning/06/images/unnamed-chunk-185-1.png)

```
## [1] 16
## attr(,"smoother")
## Call:
## loess(formula = object$oobag.improve ~ x, enp.target = min(max(4, 
##     length(x)/10), 50))
## 
## Number of Observations: 100 
## Equivalent Number of Parameters: 8.32 
## Residual Standard Error: 0.0121
```

```r
# avaliando o modelo 
confusionMatrix(data=as.factor(classe), reference=as.factor(teste$Class))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction bus opel saab van
##       bus   62    2    0   1
##       opel   0   39   23   1
##       saab   3   20   39   0
##       van    0    2    3  57
## 
## Overall Statistics
##                                              
##                Accuracy : 0.7817             
##                  95% CI : (0.7256, 0.8311)   
##     No Information Rate : 0.2579             
##     P-Value [Acc > NIR] : <0.0000000000000002
##                                              
##                   Kappa : 0.709              
##                                              
##  Mcnemar's Test P-Value : 0.1453             
## 
## Statistics by Class:
## 
##                      Class: bus Class: opel Class: saab Class: van
## Sensitivity              0.9538      0.6190      0.6000     0.9661
## Specificity              0.9840      0.8730      0.8770     0.9741
## Pos Pred Value           0.9538      0.6190      0.6290     0.9194
## Neg Pred Value           0.9840      0.8730      0.8632     0.9895
## Prevalence               0.2579      0.2500      0.2579     0.2341
## Detection Rate           0.2460      0.1548      0.1548     0.2262
## Detection Prevalence     0.2579      0.2500      0.2460     0.2460
## Balanced Accuracy        0.9689      0.7460      0.7385     0.9701
```

Note que o modelo obteve uma precisão razoável de 75,79%.

---