# FeelPath

## Set up

1. Make sure you have at least python 3.3
2. Install pipenv with `pip install pipenv` or `pip3 install pipenv` appropriately on your python version
3. set the cwd to the project
4. run `pipenv install` this will create a virtual environment installing all it needs correctly... hopefully
5. to run the API use `pipenv run python app.py`



## API usage

```{bash}
curl http://localhost:5000/ -d "data=One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The" -X POST
```

### Objects

Get object

```{json}
{
    data : "One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The"
}
```

Response object

```{ json }
{
    "emotions": [5, 4, 4],
    "next": [4.0, 3.0, 3.0]
}
```


### Bibliography

https://onlinelibrary.wiley.com/doi/10.1002/eng2.12189
