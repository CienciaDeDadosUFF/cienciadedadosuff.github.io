---
layout: post
title: "Regressão Logística"
date: 2020-09-09 23:07:41 
lang: R 
category: Modelos de Regressão
description: "Criação: Maqueise Pinheiro e Thaís Machado. Orientação: Douglas Rodrigues (UFF) e Karina Yaginuma (UFF). Colaboradores: Gabriel Miranda e Thiago Augusto."
---

### Modelos Lineares Generalizados

Os modelos lineares generalizados (MLG) são uma classe de modelos que podem ser vistos como uma extensão dos modelos de regressão linear, que torna possível modelar variáveis respostas que assumem a forma de contagem, binária, categórica e contínua assimétrica. Além disso, a suposição de normalidade não precisa ser satisfeita, pois os MLG's relacionam a distribuição aleatória da variável dependente no experimento com as outras variáveis através de uma função chamada **função de ligação**. 

Em suma, as componentes de um MLG são as seguintes:

- **Componente aleatória:** a variável de interesse a ser predita (variável resposta). É importante ressaltar que, por mais que o modelo não exija normalidade para sua distribuição, é necessário que a distribuição da variável resposta pertença a família exponencial.

- **Componente sistemática:** as $k$ variáveis explicativas associadas à variável resposta, que entram na forma de uma combinação linear de seus efeitos $$\eta_i = \sum\limits_{j=1}^{k} x_{ij} \beta_j,$$
para $i = 1, ..., n$, onde $x_{ij}$ é a i-ésima observação da j-ésima variável explicativa.

- **Função de ligação:** uma função que relaciona a componente aleatória à sistemática. Ela deve ser monótona e diferenciável. Denota-se essa função por $$g(\mu_i) = \eta_i \Rightarrow \mu_i = g^{-1}(\eta_i).$$
 
Como comentado anteriormente, os MLG's são uma **classe** de modelos, ou seja, ele é um conjunto de modelos, onde cada um é utilizado para um propósito distinto. Os modelos que o compõem são os seguintes:

- **Regressão Logística:** utilizado quando a variável resposta é binária, ou seja, quando queremos modelar uma variável categórica com 2 categorias;

- **Regressão Logística Multinomial:** utilizado quando a variável resposta é categórica com mais de 2 categorias;

- **Regressão de Poisson:** utilizado quando a variável resposta é associada a um procedimento de contagem. Isso indica que ela assume valores apenas no conjunto dos números naturais;

- **Regressão Gama:** utilizado quando a variável resposta possui distribuição assimétrica positiva.

Neste capítulo iremos focar apenas no 1° modelo, a regressão logística, por ser o mais conhecido e utilizado.

#### Regressão Logística

A regressão logística é utilizada quando a variável resposta é binária (assume valores 0 ou 1), ou seja, utilizamos quando queremos modelar uma variável categórica com 2 categorias. Vamos ver como o método funciona utilizando a base de dados a seguir. Ela possui informações sobre 777 diferentes universidades e faculdades dos EUA. Ela apresenta algumas variáveis como: `Apps` - número de pedidos recebidos para ingresso, `Enroll` - número de matrículas realizadas, `Room.Board` - custos de acomodação e alimentação, `Books` - custos estimados de livros, `PhD` - quantidade de professores com doutorado, entre outras, e nossa variável de interesse `Private`, que indica se a universidade é privada ou pública.


```r
college = readr::read_csv2("College.csv")
head(college)
```

```
## # A tibble: 6 x 18
##   Private  Apps Accept Enroll Top10perc Top25perc F.Undergrad P.Undergrad
##   <chr>   <dbl>  <dbl>  <dbl>     <dbl>     <dbl>       <dbl>       <dbl>
## 1 Yes      1660   1232    721        23        52        2885         537
## 2 Yes      2186   1924    512        16        29        2683        1227
## 3 Yes      1428   1097    336        22        50        1036          99
## 4 Yes       417    349    137        60        89         510          63
## 5 Yes       193    146     55        16        44         249         869
## 6 Yes       587    479    158        38        62         678          41
## # ... with 10 more variables: Outstate <dbl>, Room.Board <dbl>,
## #   Books <dbl>, Personal <dbl>, PhD <dbl>, Terminal <dbl>,
## #   S.F.Ratio <dbl>, perc.alumni <dbl>, Expend <dbl>, Grad.Rate <dbl>
```

Vamos ajustar um modelo de regressão logística para os dados utilizando a variável *Enroll*, através da função `glm()`. Como argumentos devem ser passados a variável a ser descrita como a combinação linear (~) da(s) outra(s), o banco de dados onde elas se encontram e a distribuição da variável resposta. Como a variável resposta é categórica com 2 categorias, ela possui uma distribuição binomial.


```r
# Primeiramente devemos transformar a variável resposta em formato 0-1, caso ela não esteja:
college$Private = ifelse(college$Private == "Yes", 1, 0)

# Aplicando o modelo aos dados:
modelo = glm(Private~Enroll, data = college, family = "binomial")
```

A principal diferença da regressão logística para a regressão linear é que, ao invés de ajustar uma linha aos dados, ela ajusta uma função logística em formato de "S". Além disso, como a variável resposta é categórica, o objetivo agora é modelar a **probabilidade** dela assumir o valor 0 ou 1. Ou seja, a variável resposta do nosso modelo será a probabilidade da universidade ser privada. Sendo assim, o modelo ajustado será da forma: $$\operatorname{P}(U_i) = \frac{exp\{\hat{\beta}_0 + \hat{\beta}_1 X_i\}}{1 + exp\{\hat{\beta}_0 + \hat{\beta}_1 X_i\}},$$ onde $U_i$ denota o evento "a i-ésima universidade ser privada" e $X_i$ é valor da variável *Enroll* relativo à i-ésima universidade.

Outra diferença importante entre a regressão linear e a regressão logística é que a regressão linear ajusta um modelo linear de forma a minimizar a soma dos quadrados dos resíduos, já na regressão logística, não há o conceito de resíduos, então não é possível utilizá-lo para construir a curva. Ao invés disso, é utilizado a função de máxima verossimilhança. São ajustadas diversas curvas aos dados e calculado a sua verossimilhança, e aquela com a maior verossimilhança é utilizada como modelo final. Podemos visualizar a curva ajustada pelo modelo através dos comandos a seguir.


```r
# Criando uma nova base que conterá uma sequência de tamanho 100 dentro da amplitude de Enroll:
dados_modelo = data.frame(Enroll = seq(min(college$Enroll), max(college$Enroll), len = 100))
# Adicionando as probabilidades de cada elemento da sequência ser privado de acordo com o modelo:
dados_modelo$Private = predict(modelo, newdata = dados_modelo, type = "response")

# Plotando os elementos da base original:
plot(Private~Enroll, data = college, col = "red3", xlab = "Número de Matrículas",
     ylab = "Probabilidade da Universidade ser Privada", 
     main = "Ajuste do Modelo de Regressão Logística")
# Adicionando a curva de regressão logística ajustada:
lines(Private ~ Enroll, dados_modelo, col = "darkgreen", lwd = 3)
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/unnamed-chunk-308-1.png)

Vamos agora observar as principais informações sobre o ajuste do modelo de regressão logística aos dados. Vamos fazer isso através da função `summary()`.


```r
summary(modelo)
```

```
## 
## Call:
## glm(formula = Private ~ Enroll, family = "binomial", data = college)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.2143  -0.1469   0.4575   0.5683   3.7370  
## 
## Coefficients:
##               Estimate Std. Error z value            Pr(>|z|)    
## (Intercept)  2.6818131  0.1676990   15.99 <0.0000000000000002 ***
## Enroll      -0.0020940  0.0001781  -11.76 <0.0000000000000002 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 910.75  on 776  degrees of freedom
## Residual deviance: 632.12  on 775  degrees of freedom
## AIC: 636.12
## 
## Number of Fisher Scoring iterations: 5
```

***1) Deviance Residuals***

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/glm1.png)


É uma medida equivalente aos resíduos dos modelos clássicos de regressão. Ou seja, quanto menores os valores, melhor é o ajuste do modelo aos dados.

***2) Coefficients***

Em regressão linear, os valores no eixo y podem assumir qualquer número no conjunto dos reais, mas com a regressão logística os valores estão limitados ao intervalo [0,1], por conta de estarmos lidando com probabilidades. Para resolver esse problema, o eixo y da regressão logística é transformado de probabilidades para log(chances). A fórmula para converter probabilidades para chances é dada por:

$$\frac{p}{1-p} = \hbox{chances}.$$

Logo, podemos obter uma fórmula que converte probabilidades em log(chances), aplicando o logaritmo nos dois lados da equação:

$$log \left( \frac{p}{1-p} \right) = log(\hbox{chances}).$$

O lado esquerdo da equação é conhecido como **função logística**. Note que o eixo y agora varia no intervalo $(-\infty, \infty)$, pois, quando:

- $p = 0 \Rightarrow log(\hbox{chances}) = log \left( \frac{0}{1-0} \right) = log(0) \rightarrow -\infty$;

- $p = 1 \Rightarrow log(\hbox{chances}) = log \left( \frac{1}{1-1} \right) = log(1/0) \rightarrow \infty$.

De acordo com James et al., com algumas manipulações conseguimos escrever a função logística como
$$log \left( \frac{p}{1-p} \right) = \beta_0 + \beta_1 X.$$
Com essa transformação, o gráfico da função é "esticado", transformando a curva em formato de "S" em uma reta. Assim, tem-se o seguinte fato importante: apesar da regressão logística ser associada com o gráfico curvilíneo, os coeficientes são apresentados em termos de log(chances). Dessa forma, a função possui um intercepto ($\beta_0$) e um coeficiente angular ($\beta_1$). Podemos verificá-los na saída da função `summary()`.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/glm2.png)


Para o nosso exemplo, então, $\hat{\beta_0} = 2,6818$ e $\hat{\beta}_1 = -0,0021$. Assim, dada uma universidade *i*, a probabilidade dela ser privada é dada por:
$$\operatorname{P}(U_i) = \frac{exp\{2,6818 -0,0021 X_i\}}{1 + exp\{2,6818 -0,0021 X_i\}}.$$

Em Std. Error são apresentados os erros padrões de cada parâmetro estimado. Assim como em regressão linear, é desejável que eles sejam pequenos.

Para todos os parâmetros é realizado um teste de hipóteses, o Teste de Wald, com o objetivo de avaliar sua significância individual para o modelo. As hipóteses do teste são: $$H_0: \beta_k = 0 \ vs  \ \beta_k \neq 0; \ k = \{0, 1\}.$$ 
Se o teste não rejeita a hipótese nula para algum parâmetro $\beta_k$, isso significa que o efeito da variável $X_{ik}$ na variável Y é nulo. Logo, ela não é necessária para o modelo. 

Na saída do `summary()`há duas saídas relacionadas a esse teste:

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/glm3.png)


- *z value* = valor observado da estatística de teste;
- *Pr(>|z|)* = p-valor do teste.

Com base em um nível de significância de 5% rejeita-se H0 para os testes de significância individual para $\beta_0$ e $\beta_1$, pois ambos os p-valores são menores do que 0,05. Ou seja, os parâmetros são significantes para o modelo, o que implica que o intercepto e o efeito das variável Xi sobre a variável resposta são significantes.

**3) Outras Medidas**

Por último, vamos comentar sobre as últimas saídas da função:

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/glm4.png)


Essas 3 medidas são utilizadas quando o objetivo é comparar o ajuste de modelos. Ou seja, quando queremos avaliar se um modelo é melhor que outro.

- *Null deviance*: mostra o quão bem a variável resposta é explicada por um modelo que inclui apenas o intercepto. Quanto menor o seu valor, melhor a explicação.

- *Residual deviance*: mostra o quão bem a variável resposta é explicada pelo modelo ajustado. Quanto menor o seu valor, melhor para o modelo.

- AIC (*Akaike Information Criterion*): o Critério de Informação Akaike é baseado na *deviance* do modelo, adicionando uma penalização relacionada ao número de variáveis explicativas (complexidade do modelo). Ou seja, assim como o $R_{aj}^{2}$ da regressão linear, sua intenção é impedir que o acréscimo de variáveis explicativas irrelevantes dê a impressão de que o modelo foi melhor ajustado. Apesar disso, ele não possui um significado (interpretação). Deve-se escolher o modelo com o menor AIC.

##### Regressão Logística com `caret`

Vamos treinar um modelo de regressão logística usando `train()`. Vamos usar a base *"transfusion"* que contém algumas informações sobre doação de sangue de 748 voluntários.


```r
library(dplyr)
transfusion = readRDS("transfusion.rds") %>%
  select(Doacoes = Frequency..times.,
         Doou = whether.he.she.donated.blood.in.March.2007)

library(caret)
set.seed(141)
noTreino = createDataPartition(y = transfusion$Doou,
                               p = 0.7, list = F)
treino = transfusion[noTreino,]
teste = transfusion[-noTreino,]

head(transfusion)
```

```
##   Doacoes Doou
## 1      50    1
## 2      13    1
## 3      16    1
## 4      20    1
## 5      24    0
## 6       4    0
```

Vamos tentar explicar a variável "Doou", que indica se o paciente doou sangue em março de 2007 ou não (1 - doou, 0 - não doou), pela variável "Doacoes", que indica o número total de doações já realizadas pelo paciente.


```r
ggplot(treino, aes(x=Doacoes,y=Doou)) + 
  geom_point(size=2)+ theme_minimal() +
  xlab("número de doações já realizadas") +
  ylab("paciente doou sangue em março/2017")
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/unnamed-chunk-311-1.png)

Aplicamos o modelo de regressão logistica adicionando o argumento `family="binomial"` no método "glm".


```r
set.seed(171)
modelo = caret::train(Doou~Doacoes, data=treino, 
               method="glm", family="binomial")
```

Podemos plotar o modelo dessa forma:


```r
ggplot(treino, aes(x=Doacoes,y=Doou)) + 
  geom_point(size=2)+
  geom_smooth(method=glm, method.args = list(family = "binomial"),
              se=FALSE,col="4") + theme_minimal() +
  xlab("número de doações já realizadas") +
  ylab("paciente doou sangue em março/2017")
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/unnamed-chunk-313-1.png)

Avaliando nosso modelo


```r
predicao = predict(modelo, teste)
postResample(predicao, teste$Doou)
```

```
##       RMSE   Rsquared        MAE 
## 0.42742456 0.01708587 0.35408946
```

