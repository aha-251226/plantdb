import os

def clean_duplicate_nodes():
    """
    nodes 테이블에서 중복된 id를 가진 레코드만 정리합니다.
    기존 고유한 데이터는 모두 보존됩니다.
    """
    
    db_path = 'knowledge_graph.duckdb'
    
    if not os.path.exists(db_path):
        print("❌ knowledge_graph.duckdb 파일을 찾을 수 없습니다.")
        return
    
    try:
        import duckdb
        conn = duckdb.connect(db_path)
        
        print("🔍 nodes 테이블 중복 검사 중...")
        
        # nodes 테이블에서 중복된 id 찾기
        duplicates = conn.execute("""
            SELECT id, COUNT(*) as count 
            FROM nodes 
            GROUP BY id 
            HAVING count > 1
            ORDER BY count DESC
        """).fetchall()
        
        if not duplicates:
            print("✅ nodes 테이블에 중복이 없습니다!")
            conn.close()
            return
        
        print(f"📊 발견된 중복 노드 ID: {len(duplicates)}개")
        print("-" * 60)
        
        total_deleted = 0
        
        for node_id, count in duplicates:
            print(f"📄 노드 ID: {node_id}")
            print(f"   중복 개수: {count}개")
            
            # 해당 노드 ID의 모든 레코드 조회
            records = conn.execute("""
                SELECT document_id, creation_time, confidence 
                FROM nodes 
                WHERE id = ? 
                ORDER BY document_id ASC, creation_time ASC
            """, [node_id]).fetchall()
            
            # 첫 번째(가장 오래된) 레코드만 유지, 나머지 삭제
            records_to_delete = records[1:]  # 첫 번째 제외하고 모두 삭제
            keep_record = records[0]  # 첫 번째 레코드 유지
            
            print(f"   🗑️  삭제할 레코드: {len(records_to_delete)}개")
            print(f"   ✅ 유지할 레코드: document_id {keep_record[0]} (생성: {keep_record[1]})")
            
            # 중복 레코드 삭제 (id와 document_id 조건으로)
            for doc_id, creation_time, confidence in records_to_delete:
                conn.execute("""
                    DELETE FROM nodes 
                    WHERE id = ? AND document_id = ? AND creation_time = ?
                """, [node_id, doc_id, creation_time])
                
                print(f"      삭제됨: document_id {doc_id} (생성: {creation_time})")
                total_deleted += 1
            
            # 관련 edges도 정리 (중복된 document_id의 edges 삭제)
            for doc_id, creation_time, confidence in records_to_delete:
                conn.execute("""
                    DELETE FROM edges 
                    WHERE document_id = ? AND (source_node = ? OR target_node = ?)
                """, [doc_id, node_id, node_id])
            
            print()
        
        print("=" * 60)
        print(f"🎉 nodes 테이블 정리 완료!")
        print(f"📊 총 삭제된 중복 노드: {total_deleted}개")
        
        # 정리 후 상태 확인
        total_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        unique_nodes = conn.execute("SELECT COUNT(DISTINCT id) FROM nodes").fetchone()[0]
        
        print(f"📈 정리 후 nodes 테이블 상태:")
        print(f"   - 전체 노드 레코드: {total_nodes}개")
        print(f"   - 고유 노드 ID: {unique_nodes}개")
        
        if total_nodes == unique_nodes:
            print("✅ 모든 중복이 제거되었습니다!")
        else:
            print("⚠️  아직 중복이 남아있을 수 있습니다.")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def preview_node_duplicates():
    """
    nodes 테이블의 중복을 미리보기합니다.
    """
    db_path = 'knowledge_graph.duckdb'
    
    try:
        import duckdb
        conn = duckdb.connect(db_path)
        
        print("🔍 nodes 테이블 중복 미리보기...")
        print("=" * 70)
        
        duplicates = conn.execute("""
            SELECT id, COUNT(*) as count 
            FROM nodes 
            GROUP BY id 
            HAVING count > 1
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()
        
        if not duplicates:
            print("✅ nodes 테이블에 중복이 없습니다!")
            return
        
        for node_id, count in duplicates:
            print(f"\n📄 노드 ID: {node_id} ({count}개 중복)")
            
            records = conn.execute("""
                SELECT document_id, creation_time, confidence 
                FROM nodes 
                WHERE id = ? 
                ORDER BY document_id ASC
            """, [node_id]).fetchall()
            
            for i, (doc_id, creation_time, confidence) in enumerate(records):
                status = "✅ 유지" if i == 0 else "🗑️ 삭제예정"
                print(f"   {status} - document_id: {doc_id}, 생성: {creation_time}")
        
        print(f"\n📊 총 {len(duplicates)}개의 중복 노드 발견")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    print("🧹 nodes 테이블 중복 정리 도구")
    print("=" * 35)
    print("💡 기존 고유한 데이터는 모두 보존됩니다!")
    
    # 먼저 미리보기
    preview_node_duplicates()
    
    # 사용자 확인
    response = input("\nnodes 테이블의 중복만 삭제하시겠습니까? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        clean_duplicate_nodes()
    else:
        print("❌ 삭제가 취소되었습니다.")