# plot2xlsx_dash

これは[plot2xlsx_tk](https://github.com/zbwtk/plot2xlsx_tk)をDashで再構築したもので、GUI化した点と、最大シート数が5に増えてる以外に違いはありません。
***

## 必要なもの
Python>=3.10  
pandas, xlsxwriter, plotly, dash, kaleido

## 実行
```
python app.py
```

***

![1](https://user-images.githubusercontent.com/126104168/221553747-91ad4e5f-f2f3-442f-af94-4454e88778b3.png)


本来HTMLやJavaScriptで書くような部分をDashで無理やり書いていたり、Dashの「各Outputは複数のメソッドに書くことはできない」といった制約のなかで書くのに苦慮しているので、正直言って見やすいコードではないです。🥺
