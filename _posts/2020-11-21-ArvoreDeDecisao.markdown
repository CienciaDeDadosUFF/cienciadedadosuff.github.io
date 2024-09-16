---
layout: post
title: "Árvore de Decisão"
date: 2020-09-09 23:07:41 
lang: R 
category: Métodos de Treino Baseados em Árvores
description: "Criação: Maqueise Pinheiro e Thaís Machado. Orientação: Douglas Rodrigues (UFF) e Karina Yaginuma (UFF). Colaboradores: Gabriel Miranda e Thiago Augusto."
---

## Métodos de Treino Baseados em Árvores

Neste capítulo será estudado de forma mais aprofundada modelos de treinamento baseados em árvores, os quais são simples para interpretação, como: árvores de decisão e regressão, florestas aleatórias, adaboost, entre outros. O objetivo é entender o funcionamento dos mesmos, assim como os critérios que utilizam para classificarem as amostras. 

É importante deixar claro que para utilizarmos esses métodos podemos usar tanto dados numéricos quanto categóricos. Além disso, não é necessário padronizar os dados.

### Árvores de Decisão

Uma árvore de decisão, em geral, pergunta uma questão e classifica o elemento baseado na resposta. Ela utiliza os dados de cada indivíduo para criar uma regra de separação, que posteriormente será utilizada para rotular novas amostras.

As árvores de decisão podem ser aplicadas aos problemas de regressão e classificação. Primeiro vamos considerar os problemas de classificação, e depois passamos para a regressão.

#### em Classificação {#link12}

Vejamos a seguir um exemplo de árvore de decisão para um problema de classificação.


![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Ex_Arvores.png)


**Nomenclatura:**

- **Nó Raiz** ou **Raiz**: é a variável que se encontra no topo da árvore;
- **Nós Internos** ou **Nós**: são as variáveis intermediárias, que possuem tanto setas apontandas para elas como saindo delas;
- **Nós Folhas** ou **Nós Terminais** ou **Folhas**: possuem apenas setas apontadas para elas. Representam a decisão final da árvore.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Nomenclatura_Arvore.png)


No processo de construção de uma árvore de decisão é importante ressaltar que a separação dos dados deve envolver apenas duas respostas: "Sim" ou "Não". Também é preciso definir a ordem das variáveis, como a variável com que se deve começar, qual deve ser a seguinte, e assim por diante. A solução para isso é obtida através do nível de **impureza** das variáveis.

Dizemos que uma variável é **impura** quando ela não consegue separar bem os dados em uma árvore de decisão. Para calcularmos a impureza de uma variável utilizamos o **indíce Gini**, que varia entre 0 (mais puro possível) e 0,5 (mais impuro possível). 
Primeiramente calculamos o índice Gini para cada nó da variável, e em seguida obtemos o índice Gini da variável como uma média ponderada. O **índice Gini** de um nó é obtido por: $$\hbox{Gini(nó)} = 1 - {p_S}^{2} - {p_N}^{2}.$$ onde $p_S$ é a proporção de "sim" da resposta da variável de interesse e $p_N$ a proporção de "não" da resposta da variável de interesse.

O **índice Gini** da variável é dado pela média do índice Gini para os nós referentes às respostas "Sim" e "Não" ponderada pela proporção dos elementos em cada nó.

$$ \hbox{Gini(variável)} = \hbox{Gini(nó}_1) \times P_1 + \hbox{Gini(nó}_2) \times P_2 $$

onde $P_1$ é a proporção de elementos no 1º nó e $P_2$ é a proporção de elementos no 2º nó.

Vamos construir uma árvore de decisão utilizando a base SmallHeart.


```r
base = readRDS("SmallHeart.rds")
head(base)
```

```
## # A tibble: 6 x 4
##   Sex   ChestPain  Thal       HeartDisease
##   <fct> <chr>      <chr>      <chr>       
## 1 M     typical    fixed      No          
## 2 F     nontypical normal     No          
## 3 M     nontypical normal     No          
## 4 F     nontypical normal     No          
## 5 M     nontypical reversable No          
## 6 M     nontypical reversable Yes
```

Nosso objetivo é prever se um indivíduo tem ou não uma doença cardíaca (variável "HeartDisease"), baseado nas outras variáveis. As variáveis explicativas são as seguintes: 

- ***Sex***: indica o sexo do indivíduo, onde "M" = Masculino e "F" = Feminino;
- ***ChestPain***: referente ao indivíduo sentir dor no peito, onde "*typical*" = típico e "*nontypical*" = não típico;
- ***Thal***: indica se o indivíduo possui Talassemia, onde "*normal*" = não possui, "*fixed*" = talassemia irreversível e "*reversable*" = talassemia reversível.

Vamos verificar o quão bem as variáveis isoladamente são capazes de prever se o paciente possui ou não doença cardíaca. Vamos começar pela variável "Sex".


```r
summary(base$Sex)
```

```
##  F  M 
## 22 50
```

Note que temos 22 indivíduos do sexo feminino e 50 indivíduos do sexo masculino. Como a resposta de um nó da árvore deve ser "Sim" ou "Não", vamos utilizar a variável "Sex=M". 


```r
# Verificando quantos indivíduos possuem doença cardíaca de acordo com o sexo:

base %>% group_by(Sex, HeartDisease) %>% summarise(N=n())
```

```
## # A tibble: 4 x 3
## # Groups:   Sex [2]
##   Sex   HeartDisease     N
##   <fct> <chr>        <int>
## 1 F     No              20
## 2 F     Yes              2
## 3 M     No              36
## 4 M     Yes             14
```

Então a variável "Sex=M" separa os pacientes da seguinte forma:

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Arvore_Sex.png)


Note que a maioria dos pacientes com doença cardíaca terminaram na folha referente ao sexo masculino, mas a maioria dos que não possuem doença também. Já podemos ter uma ideia que essa variável não é tão boa em separar os dados, mas para averiguarmos essa hipótese vamos calcular o índice gini dela.

Primeiramente vamos calcular o índice Gini do nó "Sex = M Sim":

$$ \hbox{Gini(Sex = M Sim)} = 1- \left( \frac{14}{50} \right)^{2} - \left( \frac{36}{50} \right)^{2} = 0,403.$$

Agora vamos calcular o índice Gini do nó "Sex = M Não":

$$ \hbox{Gini(Sex = M Não)} = 1- \left( \frac{2}{22} \right)^{2} - \left( \frac{20}{22} \right)^{2} = 0,166.$$

O índice Gini da variável "Sex = M" é dado pela média do índice Gini dos nós referentes às respostas "Sim" e "Não" ponderada pela frequência dos indivíduos em cada nó.

$$ \hbox{Gini(Sex = M)} = 0,403 \times \frac{50}{72} + 0,166 \times \frac{22}{72} = 0,331.$$

Como o índice Gini da variável "Sex = M" ficou mais próximo de 0,5 do que de 0, podemos constatar que ela é uma variável com baixa pureza. Note que se tivéssemos escolhido a variável "Sex = F" o índice Gini obtido seria o mesmo, pois "Sex = F Sim" é equivalente a "Sex = M Não" e "Sex = F Não" é equivalente a "Sex = M Sim". ou seja, as contas seriam as mesmas.

Agora vamos realizar o mesmo processo para a variável "ChestPain", ou seja, vamos verificar o quão bem ela é capaz de prever se o paciente possui doença cardíaca.


```r
base %>% group_by(ChestPain) %>% summarise(N=n())
```

```
## # A tibble: 2 x 2
##   ChestPain      N
##   <chr>      <int>
## 1 nontypical    49
## 2 typical       23
```

Note que temos 23 indivíduos que sentem dor no peito tipicamente e 49 indivíduos que não sentem tipicamente. Vamos verificar quantos deles possuem doença cardíaca:


```r
base %>% group_by(ChestPain, HeartDisease) %>% summarise(N=n())
```

```
## # A tibble: 4 x 3
## # Groups:   ChestPain [2]
##   ChestPain  HeartDisease     N
##   <chr>      <chr>        <int>
## 1 nontypical No              40
## 2 nontypical Yes              9
## 3 typical    No              16
## 4 typical    Yes              7
```

Vamos considerar a variável "ChestPain = Typical". Ela separa os dados da seguinte forma:

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Arvore_Chest.png)


Note que quase metade dos pacientes que possuem dor no peito têm doença cardíaca. Dos que não sentem a dor no peito, quase $\frac{1}{4}$ apenas possui a doença.

Vamos calcular o índice Gini do nó "ChestPain = Typical Sim":

$$ \hbox{Gini(ChestPain = Typical Sim)} = 1- \left( \frac{7}{23} \right)^{2} - \left( \frac{16}{23} \right)^{2} = 0,423.$$

Agora vamos calcular o índice Gini do nó "ChestPain = Typical Não":

$$ \hbox{Gini(ChestPain = Typical Não)} = 1- \left( \frac{9}{49} \right)^{2} - \left( \frac{40}{49} \right)^{2} = 0,299.$$

O índice Gini da variável "ChestPain = Typical" é dado pela média do índice Gini dos nós referentes às respostas "Sim" e "Não" ponderada pela frequência dos indivíduos em cada nó.

$$ \hbox{Gini(ChestPain = Typical)} = 0,423 \times \frac{23}{72} + 0,299 \times \frac{49}{72} = 0,339.$$

Note que ela obteve um índice Gini um pouco maior do que a variável "Sex = M". Isso indica que a variável "Sex = M" é mais pura do que a variável "ChestPain = Typical".

Agora falta apenas obter o índice Gini da variável "Thal". Mas diferentemente das outras 2 ela não possui apenas 2 níveis, e sim 3: "normal", "fixed" e "reversable".


```r
library(dplyr)
base %>% group_by(Thal) %>% summarise(N=n())
```

```
## # A tibble: 3 x 2
##   Thal           N
##   <chr>      <int>
## 1 fixed          4
## 2 normal        52
## 3 reversable    16
```


Nesse caso vamos ter que calcular o índice Gini para todas as combinações possíveis: "Thal = normal", "Thal = fixed", "Thal = reversable", "Thal = normal ou fixed", "Thal = normal ou reversable", "Thal = fixed ou reversable". Porém note que o índice Gini da variável "Thal = normal" é equivalente ao da variável "Thal = fixed ou reversable", pois "Thal = normal Sim" é o mesmo que "Thal = fixed ou reversable Não". Da mesma forma isso vale para as variáveis "Thal = fixed" e "Thal = normal ou reversable", e "Thal = reversable" e "Thal = normal ou fixed". Com isso conseguimos economizar algumas contas.


```r
base %>% group_by(Thal, HeartDisease) %>% summarise(N=n())
```

```
## # A tibble: 6 x 3
## # Groups:   Thal [3]
##   Thal       HeartDisease     N
##   <chr>      <chr>        <int>
## 1 fixed      No               3
## 2 fixed      Yes              1
## 3 normal     No              44
## 4 normal     Yes              8
## 5 reversable No               9
## 6 reversable Yes              7
```

Vamos, primeiramente, olhar para a variável "Thal = normal". Ela separa os dados da seguinte forma:

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Arvore_Thal_normal.png)


Vamos calcular o índice Gini do nó "Thal = Normal Sim":

$$ \hbox{Gini(Thal = Normal Sim)} = 1- \left( \frac{8}{52} \right)^{2} - \left( \frac{44}{52} \right)^{2} = 0,26.$$

Agora vamos calcular o índice Gini do nó "Thal = Normal Não":

$$ \hbox{Gini(Thal = Normal Não)} = 1- \left( \frac{8}{20} \right)^{2} - \left( \frac{12}{20} \right)^{2} = 0,48.$$

Então o índice Gini da variável "Thal = Normal" fica da seguinte forma:

$$ \hbox{Gini(Thal = Normal)} = 0,26 \times \frac{52}{72} + 0,48 \times \frac{20}{72} = 0,321.$$

Agora vamos olhar para a variável "Thal = Fixed". Ela separa os dados da seguinte forma:

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Arvore_Thal_fixed.png)

Vamos calcular o índice Gini do nó "Thal = Fixed Sim":

$$ \hbox{Gini(Thal = Fixed Sim)} = 1- \left( \frac{1}{4} \right)^{2} - \left( \frac{3}{4} \right)^{2} = 0,375.$$

Agora vamos calcular o índice Gini do nó "Thal = Fixed Não":

$$ \hbox{Gini(Thal = Fixed Não)} = 1- \left( \frac{15}{68} \right)^{2} - \left( \frac{53}{68} \right)^{2} = 0,344.$$

Então o índice Gini da variável "Thal = Fixed" fica da seguinte forma:

$$ \hbox{Gini(Thal = Fixed)} = 0,375 \times \frac{4}{72} + 0,344 \times \frac{68}{72} = 0,346.$$

Por último, vamos olhar para a variável "Thal = Reversable".

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Arvore_Thal_reversable.png)

Vamos calcular o índice Gini do nó "Thal = Reversable Sim":

$$ \hbox{Gini(Thal = Reversable Sim)} = 1- \left( \frac{7}{16} \right)^{2} - \left( \frac{9}{16} \right)^{2} = 0,492.$$

Agora vamos calcular o índice Gini do nó "Thal = Reversable Não":

$$ \hbox{Gini(Thal = Reversable Não)} = 1- \left( \frac{9}{56} \right)^{2} - \left( \frac{47}{56} \right)^{2} = 0,269.$$

Então o índice Gini da variável "Thal = Reversable" fica da seguinte forma:

$$ \hbox{Gini(Thal = Reversable)} = 0,492 \times \frac{16}{72} + 0,269 \times \frac{56}{72} = 0,319$$

Resumindo, os índices Ginis de todas as variáveis são:


|Variáveis           |Índice Gini |
|:-------------------|:-----------|
|Sex = M             |0,331       |
|ChestPain = Typical |0,339       |
|Thal = Normal       |0,321       |
|Thal = Fixed        |0,346       |
|Thal = Reversable   |0,319       |

A variável "Thal = Reversable" é a que possui o menor índice Gini, portanto ela é a mais pura. Ela ficará no topo da árvore de decisão, ou seja, será o nó raiz.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Raiz_Arvore_Final.png)

O próximo passo é definir as variáveis que ficarão no nó "Thal = Reversable Sim" e "Thal = Reversable Não". Para isso temos que olhar para a base de dados com os indivíduos do grupo "Thal = Reversable Sim" e "Thal = Reversable Não", respectivamente.


```r
# Grupo de indivíduos "Thal = Reversable Sim":
base1 = base %>% filter(Thal == "reversable")
head(base1)
```

```
## # A tibble: 6 x 4
##   Sex   ChestPain  Thal       HeartDisease
##   <fct> <chr>      <chr>      <chr>       
## 1 M     nontypical reversable No          
## 2 M     nontypical reversable Yes         
## 3 M     typical    reversable No          
## 4 M     nontypical reversable No          
## 5 M     nontypical reversable Yes         
## 6 M     typical    reversable Yes
```

Agora temos que calcular o índice Gini para todas as variáveis referentes a esse grupo. A que for mais pura entrará no nó "Thal = Reversable Sim". 

Poupando os cálculos, vamos obter que o menor índice Gini é o da variável "ChestPain = Typical".


```r
# Grupo de indivíduos "Thal = Reversable Não":
base2 = base %>% filter(Thal != "reversable")
head(base2)
```

```
## # A tibble: 6 x 4
##   Sex   ChestPain  Thal   HeartDisease
##   <fct> <chr>      <chr>  <chr>       
## 1 M     typical    fixed  No          
## 2 F     nontypical normal No          
## 3 M     nontypical normal No          
## 4 F     nontypical normal No          
## 5 M     nontypical normal No          
## 6 M     typical    normal No
```

Agora calculamos também o índice Gini para todas as variáveis referentes a esse grupo. Após os cálculos necessários veremos que o menor índice Gini é o da variável "ChestPain = Nontypical".

Dessa forma, podemos dar continuidade a nossa árvore.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Arvore_de_Decisao_Final.png)

Após obtidos esses novos nós, o processo continua se repetindo, obtendo novos nós e/ou folhas para a árvore, até a construção chegar ao fim.

Pergunta: quando o processo de construção de uma árvore chega ao fim? O processo de construção pode terminar por 3 fatores:

1. Quando a pureza do nó é maior do que o de qualquer variável que adicionamos;

2. Quando atingimos folhas 100% puras (índice Gini = 0);

3. Quando o ganho ao aumentar a árvore é muito pequeno. 

O ganho ao aumentar a árvore pode ser resumido como um conjunto de atributos presentes na árvore que retornem o maior ganho de informações. Essa questão será melhor abordada posteriormente, juntamente com a questão de como podar as árvores (que está intimamente relacionada ao ganho) no capítulo [XGBoost](#link10).

#### em Regressão {#link11}

Agora iremos discutir o processo de construção de uma árvore de regressão. Em uma árvore de regressão, diferentemente de uma árvore para classificação, cada folha possui um valor numérico (ao invés de categorias como "Sim" ou "Não", como no exemplo anterior da base SmallHeart). Vejamos a seguir um exemplo de árvore de decisão para um problema de regressão.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/Ex_Arvore_Reg.png)

Esse valor numérico presente nas folhas não é nada menos que a média do valor da variável de interesse a ser prevista para os elementos que satisfazem a condição do nó. Por exemplo, na árvore de regressão acima a primeira folha dá como resultado uma eficácia de 5%: essa foi a média observada da eficácia do medicamento em pacientes com mais de 50 anos de idade. Para a segunda folha, a com eficácia de 20%: esse valor é a média da eficácia do medicamento em um indivíduo com menos de 50 anos de idade e que toma uma dosagem maior do que 29mg. O processo é o mesmo para as outras folhas.

A grande pergunta é qual valor colocar no nó como condição. Para exemplificar como funciona o processo, vamos começar com um exemplo simples:

**Ex.:** Vamos carregar o banco de dados "SmallAdvertising". Este banco possui informações sobre as vendas de um produto em 10 mercados diferentes (variável *sales*), além de orçamentos de publicidade para esse produto em cada um dos mercados para três mídias diferentes: TV, rádio e jornal (variáveis TV, radio e newspaper, respectivamente).


```r
vendas = readRDS("SmallAdvertising.rds")
vendas
```

```
## # A tibble: 11 x 4
##       TV radio newspaper sales
##    <dbl> <dbl>     <dbl> <dbl>
##  1 200.    2.6      21.2  10.6
##  2  66.1   5.8      24.2   8.6
##  3 215.   24         4    17.4
##  4  23.8  35.1      65.9   9.2
##  5  97.5   7.6       7.2   9.7
##  6 204.   32.9      46    19  
##  7 195.   47.7      52.9  22.4
##  8  67.8  36.6     114    12.5
##  9 281.   39.6      55.8  24.4
## 10  69.2  20.5      18.3  11.3
## 11 147.   23.9      19.1  14.6
```

Vamos considerar o caso em que queremos construir uma árvore de regressão para prever as vendas baseados apenas na variável TV.


```r
plot(vendas$TV, vendas$sales, pch = 19,
     xlab = "Orçamento de Publicidade do Produto para a TV",
     ylab = "Vendas do Produto",
     main = "Vendas do produto x Publicidade para a TV")
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-121-1.png)

Primeiramente é preciso definir qual valor irá entrar como condição no primeiro nó. O algoritmo realiza isso testando todos os possíveis valores de separação para os dados, e pega o que minimiza a soma dos quadrados dos resíduos. Inicialmente, como o primeiro separador, ele considera a média dos 2 menores valores da Publicidade.


```r
ordenados = sort(vendas$TV)
mean(ordenados[1:2])
```

```
## [1] 44.95
```

Então 44,95 é o primeiro valor a ser testado para a separação dos dados.


```r
plot(vendas$TV, vendas$sales, pch = 19,
     xlab = "Orçamento de Publicidade do Produto para a TV",
     ylab = "Vendas do Produto",
     main = "Vendas do produto x Publicidade para a TV"); abline(v = 44.95,
                                                                 col = "red")
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-123-1.png)

Assim, o primeiro nó será da seguinte forma: 

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/ArvReg1.png)


Para a resposta "sim" prevemos que as vendas do produto será de 9,2, o qual é o resultado da média dos valores das vendas para todos os produtos cuja publicidade foi menor do que 44,95 (ou seja, é apenas o valor do primeiro elemento). Para a resposta "Não", então a folha seguinte contém o resultado da média dos valores das vendas para todos os produtos cuja publicidade foi maior do que 44,95, o qual é de 15,05.

Note que fazendo isso teremos resíduos (diferença do valor original e do valor predito pela árvore) muito grandes. O algoritmo eleva esses resíduos ao quadrado e os soma. Esse valor é a soma dos quadrados dos resíduos considerando o nó "Publicidade para a TV < 44,95?".

Em seguida ele irá para o próximo separador: a média do segundo e do terceiro menores pontos.


```r
mean(ordenados[2:3])
```

```
## [1] 66.95
```
Então 66,95 é o segundo valor a ser testado para a separação dos dados.


```r
plot(vendas$TV, vendas$sales, pch = 19,
     xlab = "Orçamento de Publicidade do Produto para a TV",
     ylab = "Vendas do Produto",
     main = "Vendas do produto x Publicidade para a TV"); abline(v = 66.95,
                                                                 col = "red")
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-125-1.png)

Então o nó considerado será da forma "Publicidade para a TV < 66,95?".

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/ArvReg2.png)


O valor de 8,9 corresponde ao resultado da média dos valores das vendas para todos os produtos cuja publicidade foi menor do que 66,95. Então a árvore prevê esse valor de vendas para o produto que obteve uma publicidade para a TV < 66,95. O valor de 15,77 é o resultado da média dos valores das vendas para todos os produtos cuja publicidade foi maior do que 66,95. Novamente serão obtidos os resíduos dessa predição e eles serão somados.

Então o algoritmo irá para o próximo separador e irá calcular a soma dos quadrados dos resíduos da predição. Isso ocorre sucessivamente até acabarem todos os separadores possíveis para a árvore. O separador vencedor (aquele que irá para o nó raiz) é aquele com a menor soma dos quadrados dos resíduos. 

A construção dos próximos nós se dá pela mesma forma que a do nó raiz. O processo de construção da árvore termina quando:

1. Atingimos um número mínimo de observações em uma folha (usualmente é utilizado 20 observações). Não continuamos a divisão após esse número mínimo pois corremos o risco de criar uma árvore sobreajustada à amostra dada;
2. Quando o ganho ao aumentar a árvore é muito pequeno.

Agora vamos para o caso em que tenhamos mais de uma variável preditiva nos dados. Vamos considerar agora que queremos prever as vendas do produto baseado em seus orçamentos de publicidade para TV, rádio e jornal. 

Assim como anteriormente, começamos usando o orçamento para a TV para prever as vendas, e pegamos o separador com a menor soma dos quadrados dos resíduos. O melhor separador se torna um candidato para a raiz da árvore. Em seguida, focamos em utilizar o orçamento para o rádio para prever as vendas. Assim como com o orçamento para a TV, tentamos diferentes separadores para a predição e calculamos a soma dos quadrados dos resíduos em cada passo. O melhor separador se torna outro candidato para a raiz. Por último, utilizamos o orçamento para o jornal para prever as vendas, e após tentarmos diferentes separadores pegamos aquele com a menor soma dos quadrados dos resíduos também. Então comparamos a soma dos quadrados dos resíduos de todos os candidatos para a raiz, e o escolhido, novamente, é aquele com a menor soma.

Para os próximos nós o processo de construção também é equivalente ao anterior, exceto que agora nós comparamos a menor soma dos quadrados dos resíduos de cada preditor. E, novamente, quando uma folha atinge um número mínimo de observações, a árvore é finalizada.

#### Construindo árvores com o `rpart` e `rpart.plot`

Vamos construir árvores usando o pacote `rpart`[@rpart]. Como argumento da função nós passamos:

1. A variável de interesse a ser prevista em função das variáveis preditoras;
2. A base de dados onde as variáveis se encontram.

Vamos utilizar a base de dados referentes ao primeiro exemplo dado de construção de uma árvore, onde queríamos prever se um indivíduo possui doença cardíaca baseado em características dele.


```r
library(rpart)
heart_arvore = rpart(HeartDisease~., data = base)
```

Agora vamos plotar a árvore utilizando o pacote `rpart.plot`[@rpartplot].


```r
library(rpart.plot)
rpart.plot(heart_arvore)
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-127-1.png)

Observe que a árvore ficou "vazia". O que ela quer dizer com isso é: assuma "Não" sempre para o indivíduo possuir doença cardíaca, e acerte com precisão de 78%. Isso ocorre devido aos valores iniciais do comando rpart.control(), que ajusta os parâmetros da função rpart(). Os principais parâmetros do rpart.control são:

- minsplit: o número mínimo de observações que devem existir em um nó para que uma divisão seja tentada. Padrão: minsplit = 20;

- minbucket: o número mínimo de observações em qualquer folha. Padrão: minbucket = minsplit/3;

- cp (complexity parameter): o mínimo de ganho de ajuste que devemos ter em cada divisão. O principal papel desse parâmetro é economizar tempo de computação removendo as divisões que não valem a pena. Padrão: cp = 0,01;

- maxdepth: profundidade máxima da árvore (a profundidade da raiz é zero). Não pode ser maior que 30.

**Ex. 1:** Vamos ajustar os parâmetros da árvore e construí-la novamente. Vamos determinar que a profundidade da árvore seja 2, que 0 seja o número mínimo de observações em um nó e que ela seja construída mesmo que não haja ganhos em mais divisões.


```r
controle = rpart.control(minsplit=0, cp = -1, maxdepth = 2)
heart_arvore = rpart(HeartDisease~., data = base, control = controle)
rpart.plot(heart_arvore)
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-128-1.png)

Note que o nó raiz é exatamente aquele que calculamos como o mais puro, o "Thal = Reversable", que é equivalente a "Thal = Fixed ou Normal". Os nós adjacentes também foram o que obtivemos anteriormente como os mais puros. 

Cada saída do comando rpart.plot() tem um significado específico:

1. A primeira saída é a classe estimada pela árvore para as amostras que se encontram naquele nó.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/rpart1.png)


2. A segunda saída é a proporção de indivíduos na classe contrária àquela estimada na primeira saída.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/rpart2.jpg)


3. A terceira saída é a porcentagem da amostra que se encontra no atual nó.

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/rpart3.png)


**Ex. 2:** Vamos agora constuir a árvore mais completa possível, ou seja, uma árvore sobreajustada à amostra, sem restrições em sua profundidade máxima.


```r
controle = rpart.control(minsplit=0, cp = -1)
heart_arvore = rpart(HeartDisease~., data = base, control = controle)
rpart.plot(heart_arvore)
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-129-1.png)

**Ex. 3:** Vamos agora considerar 10 como o número mínimo de observações em um nó e 4 como a profundidade máxima da árvore.


```r
controle = rpart.control(minsplit=10, cp = -1, maxdepth = 4)
heart_arvore = rpart(HeartDisease~., data = base, control = controle)
rpart.plot(heart_arvore)
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-130-1.png)

Agora podemos levantar a seguinte questão: como avaliar a precisão do modelo construído? Nesse exemplo nós utilizamos toda a amostra para construir a árvore, apenas para explicar o funcionamente do `rpart`, então não temos uma amostra teste para verificar o quão bom é o modelo. Então para isso teríamos que primeiramente dividir a amostra em treino e teste, depois criar o modelo com a amostra treino e em seguida aplicá-lo na amostra teste, e então, por último, poderíamos utilizar a função `confusionMatrix()` para obtermos não só a precisão como outras medidas avaliativas do modelo, além, é claro, da matriz de confusão. No tópico abaixo essas etapas serão construídas detalhadamente.

####  Construindo árvores com `train` {#link8}

Podemos utilizar árvores de decisão/regressão como um método de treinamento para os dados através da função `train()`. Vamos fazer isso utilizando a base de dados `College`. Este banco possui informações sobre 777 diferentes universidades e faculdades dos EUA. Ela apresenta algumas variáveis como: *Apps* - número de pedidos recebidos para ingresso, *Room.Board* - custos de acomodação e alimentação, *Books* - custos estimados de livros, *PhD* - quantidade de professores com doutorado, entre outras, e nossa variável de interesse *Private*, que indica se a universidade é privada ou pública.


```r
library(readr)
college = read_csv2("College.csv")
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

Vamos, primeiramente, separar a amostra em treino e teste.


```r
library(caret)
set.seed(100)
noTreino = createDataPartition(y = college$Private, p = 0.7, list = F)
treino = college[noTreino,]
teste = college[-noTreino,]
```

Vamos treinar o modelo pelo método de árvores de decisão. Fazemos isso através do argumento "*method = rpart*" da função `train()`.


```r
set.seed(100)
modelo = caret::train(Private~., method = "rpart", data = treino)
modelo
```

```
## CART 
## 
## 545 samples
##  17 predictor
##   2 classes: 'No', 'Yes' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 545, 545, 545, 545, 545, 545, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa    
##   0.04362416  0.9081649  0.7588635
##   0.20134228  0.8721337  0.6609327
##   0.51006711  0.8325440  0.5238781
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.04362416.
```

Observe que são testados alguns valores para o cp (*complexity parameter*) e é eleito aquele com a maior taxa de acurácia. Nesse caso, o cp utilizado será o de aproximadamente 0,0436. Vamos aplicar o modelo no conjunto teste.


```r
predicao = predict(modelo, teste)

# Transformando em fator para depois construirmos a matriz de confusão:
teste$Private = as.factor(teste$Private)

# Avaliando o modelo utilizando a matriz de confusão:
confusionMatrix(predicao, teste$Private)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  No Yes
##        No   44   6
##        Yes  19 163
##                                          
##                Accuracy : 0.8922         
##                  95% CI : (0.845, 0.929) 
##     No Information Rate : 0.7284         
##     P-Value [Acc > NIR] : 0.0000000007796
##                                          
##                   Kappa : 0.7088         
##                                          
##  Mcnemar's Test P-Value : 0.0164         
##                                          
##             Sensitivity : 0.6984         
##             Specificity : 0.9645         
##          Pos Pred Value : 0.8800         
##          Neg Pred Value : 0.8956         
##              Prevalence : 0.2716         
##          Detection Rate : 0.1897         
##    Detection Prevalence : 0.2155         
##       Balanced Accuracy : 0.8315         
##                                          
##        'Positive' Class : No             
## 
```

Obtivemos uma acurácia de 0,8922, o que é razoável para um modelo que utiliza árvores.


```r
# Desenhando a árvore:
rpart.plot(modelo$finalModel)
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-135-1.png)

A limitação de utilizar as árvores através do `train()` é que o único parâmetro da árvore que pode ser alterado é o cp (*complexity parameter*).


```r
modelLookup("rpart")
```

```
##   model parameter                label forReg forClass probModel
## 1 rpart        cp Complexity Parameter   TRUE     TRUE      TRUE
```

Para alterarmos o seu valor utilizamos o comando `expand.grid()`.


```r
controle = expand.grid(.cp = 0.0001)
modelo = caret::train(Private~., method = "rpart", data = treino, tuneGrid = controle)
modelo
```

```
## CART 
## 
## 545 samples
##  17 predictor
##   2 classes: 'No', 'Yes' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 545, 545, 545, 545, 545, 545, ... 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.9108212  0.7723318
## 
## Tuning parameter 'cp' was held constant at a value of 0.0001
```

Note que com esse valor de cp a árvore fica mais profunda, pois estamos diminuindo o mínimo de ganho de ajuste que devemos ter em cada divisão.


```r
rpart.plot(modelo$finalModel)
```

![tela_0]({{ site.url }}/assets/r/courses/machine_learning/01/images/unnamed-chunk-138-1.png)

---