import os

def check_database_structure():
    """
    DuckDB 데이터베이스의 테이블 구조를 확인합니다.
    """
    
    db_path = 'knowledge_graph.duckdb'
    
    if not os.path.exists(db_path):
        print("❌ knowledge_graph.duckdb 파일을 찾을 수 없습니다.")
        return
    
    try:
        import duckdb
    except ImportError:
        print("❌ DuckDB 패키지가 설치되지 않았습니다.")
        print("💡 설치 명령: pip install duckdb")
        return
    
    try:
        conn = duckdb.connect(db_path)
        
        print("🔍 데이터베이스 구조 분석 중...")
        print("=" * 60)
        
        # 모든 테이블 목록 조회
        tables = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()
        
        print(f"📊 발견된 테이블: {len(tables)}개")
        
        for (table_name,) in tables:
            print(f"\n📋 테이블: {table_name}")
            print("-" * 40)
            
            # 테이블 구조 조회
            columns = conn.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).fetchall()
            
            print("컬럼 정보:")
            for col_name, data_type, is_nullable in columns:
                nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                print(f"   📄 {col_name} ({data_type}) {nullable}")
            
            # 샘플 데이터 조회 (처음 3개)
            try:
                sample_data = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
                if sample_data:
                    print(f"\n샘플 데이터 ({len(sample_data)}개):")
                    col_names = [desc[0] for desc in conn.description]
                    
                    for i, row in enumerate(sample_data, 1):
                        print(f"   레코드 {i}:")
                        for j, value in enumerate(row):
                            # 긴 값은 잘라서 표시
                            display_value = str(value)[:50] + "..." if len(str(value)) > 50 else value
                            print(f"      {col_names[j]}: {display_value}")
                        print()
                else:
                    print("   (데이터 없음)")
            except Exception as e:
                print(f"   샘플 데이터 조회 실패: {e}")
        
        print("=" * 60)
        print("🎯 중복 검사 가능한 컬럼 찾기...")
        
        # documents 테이블이 있는지 확인
        doc_tables = [t[0] for t in tables if 'doc' in t[0].lower()]
        if doc_tables:
            for table in doc_tables:
                print(f"\n📄 {table} 테이블의 중복 가능 컬럼:")
                columns = conn.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                """).fetchall()
                
                potential_id_cols = [col[0] for col in columns if 'id' in col[0].lower() or 'name' in col[0].lower()]
                if potential_id_cols:
                    for col in potential_id_cols:
                        print(f"   🔑 {col}")
                else:
                    print("   ❌ ID 관련 컬럼을 찾을 수 없음")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_database_structure()