---
layout: post
title: "Regressão Linear"
date: 2020-09-09 23:07:41 
lang: R 
category: Modelos de Regressão
description: "Criação: Maqueise Pinheiro e Thaís Machado. Orientação: Douglas Rodrigues (UFF) e Karina Yaginuma (UFF). Colaboradores: Gabriel Miranda e Thiago Augusto."
---

### Modelos Lineares{#link18}

Modelos lineares, também conhecido como regressão linear, é um método estatístico que utiliza a relação existente entre duas ou mais variáveis de modo que uma delas pode ser prevista (explicada) através de uma combinação linear das outras. Essa relação entre as variáveis deve ser aproximadamente linear para que o modelo possa ser aplicado.

Para vermos o passo-a-passo de como o método funciona, considere o banco de dados a seguir. Ele possui informações sobre o valor das vendas (em U.M.) de 40 empresas e os valores dos seus gastos com propaganda na mídia nacional (em U.M.) e gastos com propaganda por outros meios (em U.M.).


```r
vendas = readRDS("Vendas_Empresas.rds")
head(vendas)
```

```
##   Venda Prop_outros_meios Prop_nacional
## 1 12.85               5.6           3.8
## 2 11.55               4.1           4.8
## 3 12.78               3.7           3.6
## 4 11.19               4.8           5.2
## 5  9.00               3.4           2.9
## 6  9.34               6.1           3.4
```

Vamos construir o diagrama de dispersão entre a variável Vendas e as demais.


```r
# Pacote para plotar gráfico de pontos em 3 dimensões:
library(lattice)
cloud(Venda~Prop_outros_meios*Prop_nacional, data = vendas,
      xlab = "Prop. Outros Meios", ylab = "Prop. Mídia Nacional",
      zlab = "Vendas", main = "Vendas de 40 Empresas baseado nos Gastos com Propagandas")
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/unnamed-chunk-295-1.png)

Podemos observar que há uma relação positiva e aproximadamente linear entre as vendas e os gastos com propagandas na mídia nacional e em outros meios. Ou seja, quanto maior os gastos com propaganda da empresa, maior tende a ser o valor das suas vendas. 

Dessa forma, podemos propor o seguinte modelo para os dados:

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \varepsilon_i; \ i = 1, 2,...n.$$
onde:

- $Y_i$ = valor da variável resposta para o i-ésimo elemento;

- $X_{i1}$ = valor da 1ª variável explicativa para o i-ésimo elemento;

- $X_{i2}$ = valor da 2ª variável explicativa para o i-ésimo elemento;

- $\beta_0$ = intercepto do plano com o eixo z. Representa o valor da variável Y quando as variáveis explicativas do modelo são iguais a 0;

- $\beta_1$ = é o quanto varia o valor da variável Y quando se aumenta em 1 unidade o valor da variável $X_{i1}$;

- $\beta_2$ = é o quanto varia o valor da variável Y quando se aumenta em 1 unidade o valor da variável $X_{i2}$;

- $\varepsilon_i$ = erro aleatório do modelo para o i-ésimo elemento.

Vamos ajustar um modelo de regressão linar para os dados através da função `lm()`. Como argumento deve ser passado a variável a ser descrita como a combinação linear (~) das outras e o banco de dados onde elas se encontram.


```r
modelo = lm(Venda~Prop_outros_meios+Prop_nacional, data = vendas)
modelo
```

```
## 
## Call:
## lm(formula = Venda ~ Prop_outros_meios + Prop_nacional, data = vendas)
## 
## Coefficients:
##       (Intercept)  Prop_outros_meios      Prop_nacional  
##            0.7184             1.5217             0.8145
```

Repare que o modelo retorna a estimação dos 3 parâmetros: $\hat{\beta}_0$, $\hat{\beta}_1$ e $\hat{\beta}_2$. Eles são estimados através do **método dos mínimos quadrados (MQ)**. Esse método consiste em encontrar os valores de $\beta_0$, $\beta_1$ e $\beta_2$ que minimizam a soma dos quadrados dos erros, isto é, que minimizam $$S = \sum\limits_{i=1}^{n} \varepsilon_{i}^{2}.$$

Para outras informações sobre o ajuste do modelo devemos utilizar a função `summary()`:


```r
summary(modelo)
```

```
## 
## Call:
## lm(formula = Venda ~ Prop_outros_meios + Prop_nacional, data = vendas)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -5.5280 -1.0787  0.1309  1.2272  3.5302 
## 
## Coefficients:
##                   Estimate Std. Error t value       Pr(>|t|)    
## (Intercept)         0.7184     1.3531   0.531         0.5987    
## Prop_outros_meios   1.5217     0.1764   8.628 0.000000000218 ***
## Prop_nacional       0.8145     0.3947   2.063         0.0461 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.868 on 37 degrees of freedom
## Multiple R-squared:  0.7411,	Adjusted R-squared:  0.7271 
## F-statistic: 52.96 on 2 and 37 DF,  p-value: 0.00000000001387
```

Vamos ver o que cada informação liberada por essa função representa.

***1) Residuals***

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/unnamed-chunk-295-1.png)


Em *Residuals* são apresentadas medidas estatísticas sobre os resíduos do modelo: o mínimo, o 1º quartil, a mediana, o 3º quartil e o máximo. Os resíduos são as diferenças entre os valores estimados das vendas das empresas pelo modelo e os valores reais delas. Eles refletem a qualidade do ajuste, visto que resíduos altos indicam que o modelo não se adequou muito bem aos dados. Podemos observar melhor o comportamento dos resíduos através de um *boxplot* deles.


```r
boxplot(modelo$residuals, col = "lightblue", horizontal = T, xlab = "Resíduos do Modelo",
        main = "Boxplot: Resíduos do Modelo de Regressão Linear")
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/unnamed-chunk-298-1.png)

Note que os resíduos estão centrados aproximadamente em torno de 0 e não estão muito dispersos. Além disso, estão distribuídos de forma aproximadamente simétrica.

***2) Coefficients***

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/lm2.png)


Em *Estimate* estão os valores estimados dos parâmetros $\beta_0$, $\beta_1$ e $\beta_2$. Com isso, o modelo estimado fica da seguinte forma: $$\hat{Y}_i = 0,7184 + 1,5217 X_{i1} + 0,8145 X_{i2}$$ para i = {1, 2, ..., 40}, onde:

- $\hat{\beta}_0 = 0,7184$ é o valor estimado das vendas (em u.m.) quando os gastos com propaganda por outros meios e os gastos com propaganda na mídia nacional são iguais a zero;

- $\hat{\beta}_1 = 1,5217$ é o quanto aumenta o valor estimado das vendas ao aumentar em 1 u.m. os gastos com propaganda por outros meios;

- $\hat{\beta}_2 = 0,8145$ é o quanto aumenta o valor estimado das vendas ao aumentar em 1 u.m. os gastos com propaganda na mídia nacional.

Com essa função, para cada valor do conjunto teste de gastos com propagandas por outros meios e na mídia nacional de uma empresa, conseguimos estimar o valor de suas vendas.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/lm3.png)


Em *Std. Error* estão os erros padrões de cada parâmetro estimado. O **erro padrão** é a estimativa do desvio padrão do coeficiente, por isso é desejado um erro padrão pequeno.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/lm4.png)


Para todos os parâmetros é realizado um teste de hipóteses com o objetivo de avaliar sua significância individual para o modelo. As hipóteses do teste são: $$H_0: \beta_k = 0 \ vs  \ \beta_k \neq 0; \ k = \{0, 1, 2\}.$$ 
Se o teste não rejeita a hipótese nula para algum parâmetro $\beta_k$, isso significa que o efeito da variável $X_{ik}$ na variável Y é nulo. Logo, ela não é necessária para o modelo.
A estatística do teste tem distribuição t-Student, e, portanto, a função retorna o valor observado dessa estatística para cada teste, seguido do p-valor dele.

Para o nosso modelo obtivemos os seguintes p-valores para cada teste:

- Teste de significância individual para $\beta_0$: p-valor = 0,5987;
- Teste de significância individual para $\beta_1$: p-valor < 0,0001;
- Teste de significância individual para $\beta_2$: p-valor = 0,0461.

Com base em um nível de significância de 5% rejeita-se $H_0$ para os testes de significância individual para $\beta_1$ e $\beta_2$, pois ambos os p-valores são menores do que 0,05. Ou seja, os parâmetros são significantes para o modelo, o que implica que os efeitos das variáveis $X_{i1}$ e $X_{i2}$ sobre $Y_i$ são significantes. Em outras palavras, os gastos com propaganda por outros meios e na mídia nacional das empresas têm efeito significativo sobre os valores de suas vendas. Logo, as variáveis permanecem no nosso modelo. 

Porém, note que para o intercepto o teste de significância obteve um p-valor > 0,05, ou seja, o intercepto não é significativo para o modelo. O que isso significa? Podemos interpretar da seguinte forma: o valor das vendas das empresas quando não há gastos com propaganda não é significativo. Mas isso significa que devemos retirar o intercepto do modelo? Bem, devemos analisar: retirando o intercepto do modelo estamos dizendo que o valor das vendas das empresas quando não há gastos com propaganda é de 0 u.m.. Isso seria algo extremo, visto que, por mais que as vendas sejam muito baixas quando não há gastos com propaganda, elas ainda existem. Dessa forma, vamos optar por deixar o intercepto no modelo.

Em suma, quando não rejeitamos a hipótese nula do teste de significância individual para algum $\beta_k, k \neq 0$, a variável deve ser removida do modelo. Mas quando não rejeitamos para $\beta_0$, devemos analisar o contexto do problema para saber se faz sentido retirar o intercepto, ou optar que ele permaneça.

**3) Outras Medidas**

Por último, vamos comentar sobre as últimas saídas da função:

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/lm5.png)


***3.1) Residual standard error***

O erro padrão residual é uma estimação do desvio padrão dos erros aleatórios do modelo. Sua fórmula é dada por: $$RSE = \sqrt{\sum\limits_{i=1}^{n}\frac{e_{i}^{2}}{n-p}},$$ onde:

- **$e_i$** é o i-ésimo resíduo do modelo. 
- **n** é o número de observações na amostra;
- **p** é o número de parâmetros do modelo.

Para o nosso modelo, podemos calculá-lo da seguinte forma:


```r
sqrt(sum(modelo$residuals**2)/(40-3))
```

```
## [1] 1.868454
```

***3.2) Multiple R-squared*** **e** ***Adjusted R-squared***

Aqui temos 2 medidas de qualidade do ajuste do modelo: o coeficiente de determinação ($R^{2}$) e o coeficiente de determinação ajustado ($R_{aj}^{2}$). 

O $R^{2}$ mede o quanto da variância da variável resposta é explicada pelo modelo. Interpretando-o no sentido do contexto, $R^{2} = 0,7411 \Rightarrow$ 74,11% da variância das vendas das empresas é explicada pelo modelo ajustado utilizando os gastos com propagandas por outros meios e os gastos com propagandas na mídia nacional.

Já o $R_{aj}^{2}$ é uma versão modificada do $R^{2}$ que foi ajustada para o número de preditores no modelo. O principal problema do $R^{2}$ é o fato de que toda vez que se adiciona um preditor ao modelo, ele aumenta um pouco. Isso pode acabar provocando a ilusão de que um modelo tem um melhor ajuste simplesmente porque possui mais termos, o que nem sempre é verdade. Dessa forma, no sentido de comparação, o $R_{aj}^{2}$ é o mais indicado para se verificar a qualidade do ajuste.

***3.3) F-statistic*** **e** ***p-value***

Nesta saída é liberado informações sobre o teste de significância global dos parâmetros do modelo. Neste teste é verificado a relevância de todas as variáveis explicativas conjuntamente. As hipóteses são: $$H_0: \beta_1 = ... = \beta_p = 0 \ vs \  H_1: \beta_k \neq 0 \hbox{ para algum k} = \{1, ..., p\}.$$
A estatística de teste tem distribuição F de Snedecor, e seu valor observado é liberado na primeira saída: 52,96. Em seguida é liberado o p-valor do teste, que para o nosso exemplo foi < 0,0001. Isso significa que a um nível de significância de 5% rejeita-se $H_0$, ou seja, pelo menos um parâmetro $\beta_k$ é significativo para o modelo.

>OBS: É importante ressaltar que para a aplicação de um modelo linear em uma base de dados é necessário que algumas suposições estejam satisfeitas. Para mais informações, consulte [Suposições do Modelo de Regressão Linear](#link17).

#### Modelos lineares com a função `train()`

Treinar um modelo linear usando `train()` é bem parecido com utilizando a função `lm()`. Vamos utilizar a base "Advertising" que contém dados de investimento em milhares de dolares em publicidade por meio de televisão, radio e jornal feito por 200 mercados. Nossa variável de interesse é `sales` que indica a venda em milhares de unidades desses mercados.


```r
Advertising = readr::read_csv("Advertising.csv")

library(caret)
set.seed(95)
noTreino = createDataPartition(y = Advertising$sales, p = 0.7, list = F)
treino = Advertising[noTreino,]
teste = Advertising[-noTreino,]

head(Advertising)
```

```
## # A tibble: 6 x 4
##      TV radio newspaper sales
##   <dbl> <dbl>     <dbl> <dbl>
## 1 230.   37.8      69.2  22.1
## 2  44.5  39.3      45.1  10.4
## 3  17.2  45.9      69.3   9.3
## 4 152.   41.3      58.5  18.5
## 5 181.   10.8      58.4  12.9
## 6   8.7  48.9      75     7.2
```

Nós vamos tentar explicar a vendas pela publicidade em tv e radio. O pacote `plot3D`[@plot3d] é utilizado para plotar gráficos em três dimensões.


```r
x= treino$TV
y= treino$radio
z= treino$sales

library(plot3D)
scatter3D(x, y, z, phi = 20, theta = 50, bty = "g",
          xlab = "TV", ylab = "radio", zlab = "sales",
          pch = 20, cex = 1, ticktype = "detailed", 
          main="Vendas por marketing", colkey = FALSE)
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/unnamed-chunk-301-1.png)

Nesse gráfico, as cores estão apenas sinalizando a dispersão das vendas. Note que o comportamento dos dados parece linear. A medida que o investimento na publicidade aumenta, as vendas também aumentam. Sendo assim, ajustar um modelo de regressão linear pode ser uma boa idéia. 


```r
set.seed(92)
(modelo = caret::train(sales ~ TV+radio, data = treino, 
                       method = "lm"))
```

```
## Linear Regression 
## 
## 142 samples
##   2 predictor
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 142, 142, 142, 142, 142, 142, ... 
## Resampling results:
## 
##   RMSE      Rsquared   MAE     
##   1.597285  0.9032499  1.169382
## 
## Tuning parameter 'intercept' was held constant at a value of TRUE
```

Conseguimos obter um bom valor de $R^2$ dentro da amostra. 


```r
# obtendo estimações dos coeficientes 
modelo$finalModel$coefficients
```

```
## (Intercept)          TV       radio 
##  3.23391471  0.04461188  0.18674447
```

```r
# obtendo resíduos
head( modelo$finalModel$residuals )
```

```
##         X1         X2         X3         X4         X5         X6 
##  1.5419514 -2.1582010  0.7948392 -0.4165823 -5.5538428 -0.1243164
```

```r
# Outros valores
summary(modelo)
```

```
## 
## Call:
## lm(formula = .outcome ~ ., data = dat)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -5.5538 -0.5287  0.2113  1.0583  2.4876 
## 
## Coefficients:
##             Estimate Std. Error t value            Pr(>|t|)    
## (Intercept) 3.233915   0.319236   10.13 <0.0000000000000002 ***
## TV          0.044612   0.001560   28.59 <0.0000000000000002 ***
## radio       0.186744   0.008888   21.01 <0.0000000000000002 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.537 on 139 degrees of freedom
## Multiple R-squared:  0.9077,	Adjusted R-squared:  0.9063 
## F-statistic: 683.2 on 2 and 139 DF,  p-value: < 0.00000000000000022
```

Podemos plotar o modelo dessa forma:


```r
# objetos que auxiliam a criar o plano
grid.lines = 40 # numero de grades da superficie
x.pred = seq(min(x), max(x), length.out = grid.lines)
y.pred = seq(min(y), max(y), length.out = grid.lines)
xy = expand.grid(TV = x.pred, radio = y.pred)
z.pred = matrix(predict(modelo, newdata = xy), 
                 nrow = grid.lines, ncol = grid.lines)
fitpoints = predict(modelo)
# gráfico com o plano ajustado
scatter3D(x, y, z, pch = 20, cex = 1.3, bty = "b2", col="brown2",
          theta = 30, phi = 10, ticktype = "detailed",
          xlab = "TV", ylab = "radio", zlab = "sales", 
          surf = list(x = x.pred, y = y.pred, z = z.pred,  
                      facets = NA, fit = fitpoints, border="bisque2"))
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/images/unnamed-chunk-304-1.png)

Avaliando nosso modelo


```r
predicao = predict(modelo, teste)
postResample(predicao, teste$sales)
```

```
##      RMSE  Rsquared       MAE 
## 2.0058216 0.8831412 1.4866405
```

---