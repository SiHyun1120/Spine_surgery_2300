const mysql = require('mysql');

// MySQL 연결 설정
const connection = mysql.createConnection({
  host: 'database-1.cenewnbakysq.ap-northeast-2.rds.amazonaws.com',
  user: 'admin',
  password: '',
  database: 'makersday',
  connectionLimit: 10,
  dateStrings: 'date'
});

// 연결
connection.connect(err => {
  if (err) {
    console.error('MySQL 연결 오류:', err);
  } else {
    console.log('MySQL에 연결되었습니다');
  }
});

// query 함수가 있는 db 객체를 내보냅니다.
module.exports = {
  query: (sql, values) => {
    return new Promise((resolve, reject) => {
      connection.query(sql, values, (err, results) => {
        if (err) {
          reject(err);
        } else {
          resolve(results);
        }
      });
    });
  }
};
