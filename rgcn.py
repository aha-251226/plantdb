import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data, DataLoader
import numpy as np
import json

class RGCNManager:
    """
    R-GCN (Relational Graph Convolutional Network) 매니저 클래스
    """
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_features = {}
        self.relation_types = {}
        self.num_relations = 0
        self.feature_dim = 64
        self.hidden_dim = 32
        self.output_dim = 16
        
    def initialize_model(self, num_nodes, num_relations):
        """R-GCN 모델 초기화"""
        try:
            self.num_relations = num_relations
            self.model = RGCNModel(
                num_nodes=num_nodes,
                num_relations=num_relations,
                feature_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim
            ).to(self.device)
            return True
        except Exception as e:
            print(f"❌ R-GCN 모델 초기화 실패: {e}")
            return False
    
    def prepare_graph_data(self, graph_data):
        """그래프 데이터를 R-GCN 입력 형태로 변환"""
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            if not nodes or not edges:
                print("⚠️ 그래프 데이터가 비어있습니다.")
                return None
            
            # 노드 ID를 숫자 인덱스로 매핑
            node_to_idx = {node['id']: i for i, node in enumerate(nodes)}
            
            # 관계 타입을 숫자로 매핑
            relation_types = list(set(edge.get('relation_type', 'unknown') for edge in edges))
            self.relation_types = {rel: i for i, rel in enumerate(relation_types)}
            
            # 엣지 인덱스와 관계 타입 생성
            edge_index = []
            edge_type = []
            
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                rel_type = edge.get('relation_type', 'unknown')
                
                if source in node_to_idx and target in node_to_idx:
                    edge_index.append([node_to_idx[source], node_to_idx[target]])
                    edge_type.append(self.relation_types[rel_type])
            
            if not edge_index:
                print("⚠️ 유효한 엣지가 없습니다.")
                return None
            
            # 노드 특성 생성 (고정 차원으로 조정)
            num_nodes = len(nodes)
            
            # 노드 특성을 고정된 feature_dim으로 생성
            if num_nodes <= self.feature_dim:
                # 노드 수가 적으면 원-핫 + 패딩
                node_features = torch.zeros(num_nodes, self.feature_dim).float()
                for i in range(num_nodes):
                    node_features[i, i] = 1.0
            else:
                # 노드 수가 많으면 해시 기반 특성 생성
                node_features = torch.zeros(num_nodes, self.feature_dim).float()
                for i, node in enumerate(nodes):
                    # 노드 ID를 기반으로 해시 특성 생성
                    node_id_hash = hash(node['id']) % self.feature_dim
                    node_features[i, node_id_hash] = 1.0
                    
                    # 노드 타입이 있다면 추가 특성
                    if 'ontology_class' in node:
                        type_hash = hash(node['ontology_class']) % self.feature_dim
                        if type_hash != node_id_hash:
                            node_features[i, type_hash] = 0.5
            
            # PyTorch Geometric 데이터 형태로 변환
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_type=edge_type,
                num_nodes=num_nodes
            )
            
            return data, len(relation_types)
            
        except Exception as e:
            print(f"❌ 그래프 데이터 준비 실패: {e}")
            return None
    
    def predict(self, graph_data):
        """R-GCN을 사용하여 예측 수행"""
        try:
            print("🔮 R-GCN 예측 시작...")
            
            # 그래프 데이터 준비
            prepared_data = self.prepare_graph_data(graph_data)
            if prepared_data is None:
                return self._fallback_prediction(graph_data)
            
            data, num_relations = prepared_data
            
            # 모델 초기화
            if not self.initialize_model(data.num_nodes, num_relations):
                return self._fallback_prediction(graph_data)
            
            # 예측 수행
            self.model.eval()
            with torch.no_grad():
                data = data.to(self.device)
                embeddings = self.model(data.x, data.edge_index, data.edge_type)
                
                # 노드 임베딩을 기반으로 예측 결과 생성
                predictions = self._generate_predictions(embeddings, graph_data)
            
            print("✅ R-GCN 예측 완료!")
            return predictions
            
        except Exception as e:
            print(f"❌ R-GCN 예측 실패: {e}")
            return self._fallback_prediction(graph_data)
    
    def _generate_predictions(self, embeddings, graph_data):
        """임베딩을 기반으로 예측 결과 생성"""
        try:
            nodes = graph_data.get('nodes', [])
            predictions = {
                'node_embeddings': embeddings.cpu().numpy().tolist(),
                'predictions': [],
                'confidence_scores': [],
                'relation_recommendations': [],
                'average_confidence': 0.0,  # 추가
                'total_nodes': len(nodes),  # 추가
                'prediction_summary': {}    # 추가
            }
            
            # 각 노드에 대한 예측
            total_confidence = 0.0
            cluster_counts = {'high_importance': 0, 'medium_importance': 0, 'low_importance': 0}
            
            for i, node in enumerate(nodes):
                # 임베딩 기반 클러스터링 (간단한 예시)
                embedding = embeddings[i].cpu().numpy()
                cluster = self._simple_clustering(embedding)
                confidence = float(torch.sigmoid(torch.norm(embeddings[i])))
                
                predictions['predictions'].append({
                    'node_id': node['id'],
                    'predicted_cluster': cluster,
                    'confidence': confidence,
                    'embedding_norm': float(torch.norm(embeddings[i]))
                })
                
                predictions['confidence_scores'].append(confidence)
                total_confidence += confidence
                cluster_counts[cluster] += 1
            
            # 평균 신뢰도 계산
            predictions['average_confidence'] = total_confidence / max(1, len(nodes))
            predictions['class_distribution'] = cluster_counts  # 최상위로 이동
            predictions['prediction_summary'] = {
                'cluster_distribution': cluster_counts,
                'avg_confidence': predictions['average_confidence'],
                'high_confidence_nodes': sum(1 for c in predictions['confidence_scores'] if c > 0.7),
                'total_predictions': len(predictions['predictions'])
            }
            
            # 관계 추천 (상위 유사도 기반)
            similarities = torch.mm(embeddings, embeddings.t())
            top_pairs = torch.topk(similarities.flatten(), k=min(10, len(nodes)//2))
            
            for idx in top_pairs.indices:
                i, j = divmod(idx.item(), len(nodes))
                if i != j:  # 자기 자신 제외
                    predictions['relation_recommendations'].append({
                        'source': nodes[i]['id'],
                        'target': nodes[j]['id'],
                        'similarity': float(similarities[i, j]),
                        'recommended_relation': 'similar_to'
                    })
            
            return predictions
            
        except Exception as e:
            print(f"❌ 예측 결과 생성 실패: {e}")
            return self._fallback_prediction(graph_data)
    
    def _simple_clustering(self, embedding):
        """간단한 클러스터링 (예시)"""
        # 임베딩의 첫 번째 차원을 기준으로 클러스터링
        if embedding[0] > 0.5:
            return "high_importance"
        elif embedding[0] > 0:
            return "medium_importance"
        else:
            return "low_importance"
    
    def _fallback_prediction(self, graph_data):
        """R-GCN 실패 시 대체 예측"""
        print("🔄 대체 예측 방법 사용...")
        
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        predictions = {
            'node_embeddings': [],
            'predictions': [],
            'confidence_scores': [],
            'relation_recommendations': [],
            'fallback': True,
            'average_confidence': 0.0,      # 추가
            'total_nodes': len(nodes),      # 추가
            'prediction_summary': {}        # 추가
        }
        
        # 간단한 그래프 분석 기반 예측
        total_confidence = 0.0
        cluster_counts = {'high_importance': 0, 'medium_importance': 0, 'low_importance': 0}
        
        for node in nodes:
            # 노드의 연결 정도 기반 중요도 계산
            degree = sum(1 for edge in edges if edge.get('source') == node['id'] or edge.get('target') == node['id'])
            importance = min(degree / max(1, len(nodes) * 0.1), 1.0)
            cluster = "high_importance" if importance > 0.7 else "medium_importance" if importance > 0.3 else "low_importance"
            
            predictions['predictions'].append({
                'node_id': node['id'],
                'predicted_cluster': cluster,
                'confidence': importance,
                'degree': degree
            })
            
            predictions['confidence_scores'].append(importance)
            predictions['node_embeddings'].append([importance, degree, len(node.get('id', ''))])
            total_confidence += importance
            cluster_counts[cluster] += 1
        
        # 평균 신뢰도 및 요약 정보 추가
        predictions['average_confidence'] = total_confidence / max(1, len(nodes))
        predictions['class_distribution'] = cluster_counts  # 최상위로 이동
        predictions['prediction_summary'] = {
            'cluster_distribution': cluster_counts,
            'avg_confidence': predictions['average_confidence'],
            'high_confidence_nodes': sum(1 for c in predictions['confidence_scores'] if c > 0.7),
            'total_predictions': len(predictions['predictions']),
            'fallback_method': True
        }
        
        return predictions
    
    def get_status(self):
        """R-GCN 매니저의 현재 상태 반환"""
        try:
            status = {
                'initialized': self.model is not None,
                'device': str(self.device),
                'num_relations': self.num_relations,
                'feature_dim': self.feature_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'available': True,
                'torch_available': torch.cuda.is_available(),
                'relation_types': len(self.relation_types)
            }
            return status
        except Exception as e:
            return {
                'initialized': False,
                'available': False,
                'error': str(e),
                'device': 'unknown'
            }


class RGCNModel(torch.nn.Module):
    """R-GCN 모델 클래스"""
    
    def __init__(self, num_nodes, num_relations, feature_dim, hidden_dim, output_dim):
        super(RGCNModel, self).__init__()
        
        self.num_relations = num_relations
        self.conv1 = RGCNConv(feature_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations)
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_type):
        # 첫 번째 R-GCN 층
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 두 번째 R-GCN 층
        x = self.conv2(x, edge_index, edge_type)
        
        return x