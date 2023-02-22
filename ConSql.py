import pymysql as mysql
# conn=mysql.Connection(host="localhost",user="root",password="xfblackzero20012",database="car_stop_place",port=3306)
# cursor=conn.cursor()
# conn.select_db("car_stop_place")
# cursor.execute("create table time_info(id VARCHAR(20),inTime DATETIME,outTime DATETIME)")
# conn.autocommit(True)
# ret=cursor.execute("insert into time_info(id,inTime,outTime)values('湘A00000',NOW(),NULL)")
# ret=cursor.execute("update time_info set outTime=NOW() where id='湘A00000'")
# ret=cursor.execute("select * from time_info where id='湘A00000'")
# row:tuple=cursor.fetchone()
# print(f"id={row[0]},type={type(row[1])},inTime={row[1]},outTime={row[2]},try={row[2]-row[1]}")
def initConnect(DB_name:str):
    conn = mysql.Connection(host="localhost", user="root", password="xfblackzero20012",
                            database="car_stop_place",port=3306)
    conn.select_db("car_stop_place")
    conn.autocommit(True)
    return conn

conn = initConnect("car_stop_place")
cursor = conn.cursor()

def initPlaceState():

    for i in range(10):
        cursor.execute(f"insert into place_state(no,id,count,time)values({i+1},NULL,0,NULL)")
    cursor.close()
    conn.close();

def selectAllPlace()->tuple:
    cursor.execute(f"select * from place_state")
    rows:tuple=cursor.fetchall();
    # print(f"type{type(rows)},rows={rows}")
    for row in rows:
        print(row)

    return rows


if __name__=="__main__":
   # initPlaceState()
    selectAllPlace();
    cursor.close()
    conn.close();