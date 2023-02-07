'use strict';

const express = require('express');

// константы
const PORT = 80;
const HOST = '0.0.0.0';

// приложение
const app = express();
app.get('/', (req, res) => {
  res.send('Hello World');
});

app.listen(PORT, HOST, () => {
  console.log(`Running on http://${HOST}:${PORT}`);
});