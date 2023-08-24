const session = require('express-session');
const express = require('express');
const app = express();
const db = require('./db');
const bcrypt = require('bcrypt');
const path = require('path');
const axios = require('axios');
const bodyParser = require('body-parser');


// 세션 설정
app.use(session({
  secret: 'random-secret-key-here', // 세션을 암호화하기 위한 비밀키
  resave: false,
  saveUninitialized: true
}));


const PORT = process.env.PORT || 5000

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(bodyParser.json());

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'main.html'));
});

app.get('/main', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'main.html'));
});

app.get('/login', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'login.html'));
});

app.get('/regis', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'regis.html'));
});

app.post('/register', async (req, res) => {
  const { username, email, password, passwordcheck } = req.body;

  try {

    // 아이디에 하나 이상의 문자가 있는지 검사
    if (!/[a-zA-Z]/.test(username)) {
      throw new Error('아이디에는 하나 이상의 문자가 포함되어야 합니다');
    }

    // 비밀번호에 하나 이상의 숫자가 있는지 검사
    if (!/\d/.test(password)) {
      throw new Error('비밀번호에는 하나 이상의 숫자가 포함되어야 합니다');
    }

    // 비밀번호와 비밀번호 확인이 일치하는지 검사
    if (password !== passwordcheck) {
      throw new Error('비밀번호와 비밀번호 확인이 일치하지 않습니다');
    }

    // 비밀번호 해시
    const hashedPassword = await bcrypt.hash(password, 10);

    // 아이디 중복 여부 확인
    const userCountResult = await db.query(
      'SELECT COUNT(*) AS count FROM posturetech WHERE id = ?',
      [username]
    );
    const userCount = userCountResult[0].count;

    if (userCount > 0) {
      throw new Error('이미 사용 중인 아이디입니다');
    }

    // 중복이 아닐 경우 등록 처리
    await db.query(
      'INSERT INTO posturetech (id, email, password, passwordcheck) VALUES (?, ?, ?, ?)',
      [username, email || null, hashedPassword, passwordcheck]
    );


    const successMessage = '사용자가 성공적으로 등록되었습니다';
    console.log(successMessage);
    res.send('<script>alert("' + successMessage + '"); window.location.href="/login";</script>');
  } catch (error) {
    console.error(error);
    res.send('<script>alert("' + error.message + '"); window.location.href="/regis";</script>');
  }
});

app.post('/loginpage', async (req, res) => {
  const { username, password } = req.body;

  try {
    // 아이디에 하나 이상의 문자가 있는지 검사
    if (!/[a-zA-Z]/.test(username)) {
      throw new Error('아이디에는 하나 이상의 문자가 포함되어야 합니다');
    }
    
    // 비밀번호에 하나 이상의 숫자가 있는지 검사
    if (!/\d/.test(password)) {
      throw new Error('비밀번호에는 하나 이상의 숫자가 포함되어야 합니다');
    }

    // 사용자 정보 데이터베이스에서 조회
    const userResult = await db.query(
      'SELECT id, password FROM posturetech WHERE id = ?',
      [username]
    );
    
    const user = userResult[0];

    if (!user) {
      throw new Error('해당 아이디로 등록된 사용자가 없습니다');
    }

    // 저장된 해시된 비밀번호와 입력한 비밀번호를 비교
    const passwordMatch = await bcrypt.compare(password, user.password);


    if (!passwordMatch) {
      throw new Error('비밀번호가 일치하지 않습니다');
    }

    // 로그인 성공 시 처리
    console.log('로그인 성공:', username);

    // 로그인된 사용자 아이디를 세션에 저장
    req.session.username = username;

    res.send('<script>alert("로그인 성공!"); window.location.href="/main.html";</script>');

    axios.post('http://192.168.0.13:3000/send-username', {username})
      .then(response => {
        console.log(response.data.message)
      })
      .catch(error => {
        console.error('Error sending data:', error);
      });

  } catch (error) {
    console.error('로그인 오류:', error.message);
    res.send('<script>alert("' + error.message + '"); window.location.href="/login";</script>');
  }
});


app.post('/save-data', async (req, res) => {
    const { username, turtleCount, spineCount, date } = req.body;

    try {
        // 데이터베이스에 데이터 저장
        await db.query(
            'INSERT INTO postureUser (username, textneck, spine, date) VALUES (?, ?, ?, ?)',
            [username, turtleCount, spineCount, date]
        );
        console.log('mysql 저장완료');

        res.json({ message: 'Data saved successfully.' });
    } catch (error) {
        console.error('Error saving data:', error);
        res.status(500).json({ message: 'An error occurred while saving data.' });
    }
});

app.get('/data-graph', async (req, res) => {
  const username = req.session.username;

  try {
    const userResults = await db.query(
      'SELECT date, textneck, spine FROM postureUser WHERE username = ?',
      [username]
    );

    console.log('Fetched userResults:', userResults); // 추가된 부분

    const dates = [];
    const textneckCounts = [];
    const spineCounts = [];

    userResults.forEach(result => {
      dates.push(result.date);
      textneckCounts.push(result.textneck);
      spineCounts.push(result.spine);
    });

    console.log('Fetched data:', { dates, textneckCounts, spineCounts }); // 추가된 부분

    res.json({ dates, textneckCounts, spineCounts });
  } catch (error) {
    console.error('Error fetching user data:', error);
    res.status(500).json({ message: 'An error occurred while fetching user data.' });
  }
});


app.listen(PORT, () => {
  console.log(`서버가 포트 ${PORT}에서 실행 중입니다`);
});
