var XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest;

var url = "http://localhost:5000/";

var xhr = new XMLHttpRequest();
xhr.open("POST", url);

xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");

xhr.onreadystatechange = function () {
   if (xhr.readyState === 4) {
      console.log(xhr.status);
      console.log(xhr.responseText);
   }};

var data = "data=Remember the milk. or juice";

xhr.send(data);
