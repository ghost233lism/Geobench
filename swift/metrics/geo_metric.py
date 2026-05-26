# Copyright (c) Alibaba, Inc. and its affiliates.
import time
import json
import re
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
from transformers.trainer_utils import EvalPrediction
from geopy.distance import geodesic
from tqdm import tqdm
import time
import math
from swift.utils import Serializer, get_current_device, get_logger

logger = get_logger()


class BaiduGeocoding:
    """百度地理编码功能集成"""
    
    def __init__(self, api_key: str = "ROEabeCOaLRry8eoNnXECmzoELA5dJl2"):
        self.api_key = api_key
        self.base_url = "https://api.map.baidu.com"
        
    def geocode(self, address: str) -> Optional[tuple]:
        """
        地址转坐标（地理编码）
        
        Args:
            address: 要解析的地址
            
        Returns:
            (lat, lon) 坐标元组，失败返回None
        """
        url = f"{self.base_url}/geocoding/v3/"
        params = {
            "address": address,
            "output": "json",
            "ak": self.api_key,
            "extension_analys_level": "1",
            "ret_coordtype": "bd09ll"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == 0:
                location = result["result"]["location"]
                return location['lat'], location['lng']
            else:
                logger.warning(f"地理编码失败: {result.get('message', '未知错误')}")
                return None
        except Exception as e:
            logger.error(f"地理编码请求失败: {str(e)}")
            return None


class OpenCageGeocoding:
    """OpenCage地理编码功能集成"""
    
    def __init__(self, api_key_list=['7d2eca3fd1e14204a00cb1c1c354d183','fb3f0e5f032b414891c1e696b9ba66c0','07b868837b67445a997bd6f942e0a09f','d48d2940895c4f178eb0ee6fd31193eb'
        ],language: str = "en"):
        """
        初始化OpenCage地理编码器
        
        Args:
            api_key: OpenCage API密钥
            language: 返回结果的语言，默认为英文
        """
        self.api_key_list = api_key_list
        self.base_url = "https://api.opencagedata.com/geocode/v1/json"
        self.language = language
        
    def geocode_old(self, address: str) -> Optional[tuple]:
        """
        地址转坐标（地理编码）
        
        Args:
            address: 要解析的地址
            
        Returns:
            (lat, lng) 坐标元组，失败返回None
        """
        params = {
            'q': address,
            'key': self.api_key,
            'language': self.language,
            'limit': 1,
            'no_annotations': 1 
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('results'):
                result = data['results'][0]
                coordinates = result['geometry']
                lat = coordinates['lat']
                lng = coordinates['lng']
                return lat, lng
            else:
                # logger.warning(f"OpenCage地理编码失败: 未找到匹配的地址")
                return None
        except Exception as e:
            logger.error(f"OpenCage地理编码请求失败: {str(e)}")
            return None
    
    def geocode(self, address: str, retry_count: int = 0) -> Dict:
        """
        调用OpenCage地理编码API
        
        Args:
            address: 要编码的地址
            retry_count: 重试次数
            
        Returns:
            包含坐标和置信度信息的字典
        """
        import random
        random_num = random.randint(1, len(self.api_key_list))
        api_key=self.api_key_list[random_num-1]

        
        
        params = {
            'q': address.replace(';', ',').replace('；', ',').replace('，', ','),
            'key': api_key,
            'language': 'en',  # 英文结果
            'limit': 1,
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            


            #成功查询到结果
            if data["status"]["code"]==200:
                results=data["results"]
                if len(results)==0:
                    print(f"[GeoScoreAccuracy] 没有查询到结果，message:{data['status']['message']}, address:{address}")
                    # 尝试去掉最后一个地址组件重试
                    address_parts = address.split(';')
                    if len(address_parts) > 1:
                        # 移除最后一部分，重新组合
                        shorter_address = ';'.join(address_parts[:-1])
                        return self.geocode(shorter_address, retry_count)
                    return None
                else:
                    results=results[0]
                    # 提取坐标
                    coordinates = results['geometry']
                    lat = coordinates['lat']
                    lng = coordinates['lng']
                
                    # 提取地址组件
                    components = results.get('components', {})
                    formatted_address = results.get('formatted', '')
                    confidence = results.get('confidence', 0)
                
                    api_result = {
                        "status": "success",
                        "longitude": lng,
                        "latitude": lat,
                        "confidence": confidence,
                        "formatted_address": formatted_address,
                        "components": components
                    }
                
                    return lat,lng

            #速率限制
            if data["status"]["code"]==429:
                print(f"[GeoScoreAccuracy] 速率限制: {data['status']['message']}")
                if retry_count < 5:
                    print(f"[GeoScoreAccuracy] 速率限制，第{retry_count+1}次重试")
                    time.sleep(0.1)
                    return self.geocode(address, retry_count + 1)
                else:
                    return None
            #地址太长
            if data["status"]["code"]==410:
                print(f"[GeoScoreAccuracy] 地址太长: message:{data['status']['message']}, address:{address}")
                # 尝试去掉最后一个地址组件重试
                address_parts = address.split(';')
                if len(address_parts) > 1:
                    # 移除最后一部分，重新组合
                    shorter_address = ';'.join(address_parts[:-1])
                    return self.geocode(shorter_address, retry_count)
                else:
                    return None
            #超时
            if data["status"]["code"]==408:
                print(f"[GeoScoreAccuracy] 超时: message:{data['status']['message']}, address:{address}")
                if retry_count < 5:
                    print(f"[GeoScoreAccuracy] 超时，第{retry_count+1}次重试")
                    time.sleep(0.1)
                    return self.geocode(address, retry_count + 1)
                else:
                    return None

            #禁止
            if data["status"]["code"]==403:
                print(f"[GeoScoreAccuracy] 禁止: message:{data['status']['message']}, address:{address}")
                return None

            #其他错误
            else:
                print(f"[GeoScoreAccuracy] OpenCage API其他错误: {data['status']}")
                return None

                
        except requests.exceptions.RequestException as e:
            print(f"[GeoScoreAccuracy] OpenCage API请求时发生错误: {e}")
            
            # 网络错误可以重试
            if retry_count < 5:
                print(f"[GeoScoreAccuracy] OpenCage API请求时发生错误，第{retry_count+1}次重试")
                time.sleep(0.1)
                return self.geocode(address, retry_count + 1)
            else:
                return None
        except Exception as e:
            return None
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        计算两个经纬度点之间的距离（公里）
        使用Haversine公式
        
        Args:
            lat1, lon1: 第一个点的纬度和经度
            lat2, lon2: 第二个点的纬度和经度
            
        Returns:
            两点之间的距离（公里）
        """
        # 将角度转换为弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # 地球半径（公里）
        r = 6371
        return c * r
    

class Metric(ABC):

    def __init__(self):
        self._default = {}
        self._default_factory = {}

    def add_state(self, name: str, default=None, default_factory=None) -> None:
        if not hasattr(self, '_default'):
            raise AttributeError('Please call super().__init__() first.')
        if default is None:
            self._default_factory[name] = default_factory
            assert name not in self._default, f'self._default: {self._default}'
            default = default_factory()
        else:
            self._default[name] = default
            assert name not in self._default_factory, f'self._default_factory: {self._default_factory}'
        setattr(self, name, default)

    def reset(self):
        for k, v in self._default.items():
            setattr(self, k, v)
        for k, v in self._default_factory.items():
            setattr(self, k, v())

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass


class InferStats(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('start_runtime', default_factory=lambda: time.perf_counter())
        self.add_state('num_prompt_tokens', default_factory=dict)
        self.add_state('num_generated_tokens', default_factory=dict)

    def update(self, output):
        id_ = output.id
        self.num_prompt_tokens[id_] = output.usage.prompt_tokens
        self.num_generated_tokens[id_] = output.usage.completion_tokens

    def compute(self):
        runtime = time.perf_counter() - self.start_runtime
        num_samples = len(self.num_generated_tokens)
        num_generated_tokens = sum(self.num_generated_tokens.values())
        return {
            'num_prompt_tokens': sum(self.num_prompt_tokens.values()),
            'num_generated_tokens': num_generated_tokens,
            'num_samples': num_samples,
            'runtime': runtime,
            'samples/s': num_samples / runtime,
            'tokens/s': num_generated_tokens / runtime,
        }


class MeanMetric(Metric):

    def __init__(self, nan_value=0, device=None):
        super().__init__()
        self.nan_value = nan_value
        self.add_state('state', default=0.)
        self.add_state('count', default=0)
        if device is None:
            device = get_current_device()
        self.device = device

    def update(self, state: torch.Tensor):
        if isinstance(state, (torch.Tensor, np.ndarray)):
            count = len(state)
            state = state.sum().item()
        elif isinstance(state, (list, tuple)):
            count = len(state)
            state = sum(state)
        else:
            count = 1

        self.state += state
        self.count += count

    def compute(self):
        if self.count == 0:
            value = self.nan_value
        elif dist.is_initialized():
            tensor = torch.tensor([self.state, self.count], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            value = (tensor[0] / tensor[1]).item()
        else:
            value = self.state / self.count
        return {
            'value': value,
        }


class GeometricMetricAgent(Metric):
    """地理位置度量计算器"""
    
    def __init__(self, max_distance: float = 18050.0):
        super().__init__()
        self.max_distance = max_distance
        self.geocoder = OpenCageGeocoding()
        self.add_state('distances', default_factory=list)
        self.add_state('valid_predictions', default=0)
        self.add_state('total_samples', default=0)
        self.add_state('predicted_coordinates', default_factory=list)
        self.add_state('individual_geoscores', default_factory=list)
    
    def _extract_answer_from_json(self, response: str) -> Optional[str]:
        """从JSON响应中提取answer字段"""
        try:
            # 尝试直接解析JSON
            if response.strip().startswith('{'):
                data = json.loads(response)
                if 'answer' in data:
                    return data['answer']
                elif 'FinalAnswer' in data:
                    return data['FinalAnswer']
            
            # 尝试从文本中提取JSON
            json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('answer', '')
                
            # 尝试提取Final answer
            final_answer_match = re.search(r'"FinalAnswer":\s*"([^"]*)"', response)
            if final_answer_match:
                return final_answer_match.group(1)
                
            logger.warning(f"无法从响应中提取answer: {response[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"提取answer失败: {e}")
            return None
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两个坐标点之间的距离（公里）"""
        try:
            return geodesic((lat1, lon1), (lat2, lon2)).kilometers
        except Exception as e:
            logger.error(f"计算距离失败: {e}")
            return float('inf')
    
    def update(self, predictions: List[str], labels: List[Dict]):
        """
        更新度量状态
        
        Args:
            predictions: 模型预测的JSON字符串列表
            labels: 包含真实坐标的标签列表，格式如 [{'lat_bd': x, 'lon_bd': y}, ...]
        """
        for pred, label in tqdm(zip(predictions, labels), total=len(predictions), desc="Processing predictions"):
            self.total_samples += 1
            
            # 提取answer字段
            answer = self._extract_answer_from_json(pred)
            if not answer:
                self.distances.append(float('inf'))
                self.predicted_coordinates.append(None)
                self.individual_geoscores.append(0.0)
                continue
            
            # 地理编码获取坐标，若失败，则逐步去掉最后一个分号分隔的部分再试
            coords = self.geocoder.geocode(answer)
            temp_answer = answer
            if coords is None:
                # 允许最多去掉两个分号后面的部分
                for step in range(2):
                    if ';' not in temp_answer:
                        break
                    parts = [part.strip() for part in temp_answer.split(';')]
                    if len(parts) <= 1:
                        break
                    # 若只剩下两个部分，只去一次
                    if len(parts) == 2:
                        temp_answer = '; '.join(parts[:-1])
                        coords = self.geocoder.geocode(temp_answer)
                        break
                    temp_answer = '; '.join(parts[:-1])
                    coords = self.geocoder.geocode(temp_answer)
                    if coords is not None:
                        break
            
            # 如果通过去掉分号获得了坐标，检查与真实坐标的距离
            if coords is not None and temp_answer != answer:
                pred_lat, pred_lon = coords
                true_lat = label.get('lat_bd', label.get('lat', 0))
                true_lon = label.get('lon_bd', label.get('lon', 0))
                distance_check = self._calculate_distance(true_lat, true_lon, pred_lat, pred_lon)
                if distance_check > 250:
                    coords = None
            if coords is None:
                self.distances.append(float('inf'))
                self.predicted_coordinates.append(None)
                self.individual_geoscores.append(0.0)
                continue
            
            pred_lat, pred_lon = coords
            # 优先使用lat_bd/lon_bd, 获取不到则尝试lat/lon，最后为0
            true_lat = label.get('lat_bd', label.get('lat', 0))
            true_lon = label.get('lon_bd', label.get('lon', 0))
            
            # 保存预测坐标
            self.predicted_coordinates.append({
                'lat': pred_lat,
                'lon': pred_lon,
                'address': answer
            })
            
            # 计算距离
            distance = self._calculate_distance(true_lat, true_lon, pred_lat, pred_lon)
            self.distances.append(distance)
            
            # 计算单个geoscore
            if distance != float('inf'):
                geoscore = 5000 * np.exp(-10 * (distance / self.max_distance))
                self.valid_predictions += 1
            else:
                geoscore = 0.0
            
            self.individual_geoscores.append(geoscore)
            
            # 避免API限制
            time.sleep(0.1)
    
    def compute(self):
        """计算最终的地理度量指标"""
        if not self.distances:
            return {}
        
        valid_distances = [d for d in self.distances if d != float('inf')]
        if not valid_distances:
            return {
                'mean_distance': 0.0,
                'median_distance': 0.0,
                'continent_accuracy': 0.0,
                'country_accuracy': 0.0,
                'region_accuracy': 0.0,
                'city_accuracy': 0.0,
                'street_accuracy': 0.0,
                'mean_geoscore': 0.0,
                'median_geoscore': 0.0,
                'valid_predictions': 0,
                'total_samples': self.total_samples
            }
        
        # 仅对有效样本（valid_distances）进行指标计算，无效样本全部排除，不纳入计算
        distances_array = np.array(valid_distances)
        num_valid = len(distances_array)

        # 基础距离指标
        if num_valid > 0:
            mean_distance = np.mean(distances_array)
            median_distance = np.median(distances_array)

            # 准确率指标
            continent_acc = np.mean(distances_array < 2500) * 100  # < 2500km
            country_acc = np.mean(distances_array < 750) * 100  # < 750km
            region_acc = np.mean(distances_array < 200) * 100  # < 200km
            city_acc = np.mean(distances_array < 25) * 100      # < 25km
            street_acc = np.mean(distances_array < 1) * 100     # < 1km
        else:
            mean_distance = 0.0
            median_distance = 0.0
            continent_acc = 0.0
            country_acc = 0.0
            region_acc = 0.0
            city_acc = 0.0
            street_acc = 0.0
        
        # GeoScore计算（只使用有效预测）
        valid_geoscores = [score for score, dist in zip(self.individual_geoscores, self.distances) if dist != float('inf')]
        if valid_geoscores:
            mean_geoscore = np.mean(valid_geoscores)
            median_geoscore = np.median(valid_geoscores)
        else:
            mean_geoscore = 0.0
            median_geoscore = 0.0
        
        return {
            'mean_distance': float(mean_distance),
            'median_distance': float(median_distance),
            'continent_accuracy': float(continent_acc),
            'country_accuracy': float(country_acc),
            'region_accuracy': float(region_acc),
            'city_accuracy': float(city_acc),
            'street_accuracy': float(street_acc),
            'mean_geoscore': float(mean_geoscore),
            'median_geoscore': float(median_geoscore),
            'valid_predictions': self.valid_predictions,
            'total_samples': self.total_samples
        }
    
    def get_predicted_coordinates(self) -> List[Optional[Dict]]:
        """
        获取所有预测的坐标信息
        
        Returns:
            坐标信息列表，每个元素是字典包含lat, lon, address，失败的预测为None
        """
        return self.predicted_coordinates.copy()
    
    def get_individual_metrics(self) -> List[Dict]:
        """
        获取每个数据的距离和geoscore
        
        Returns:
            每个数据的指标列表，包含distance和geoscore
        """
        return [
            {
                'distance': dist if dist != float('inf') else None,
                'geoscore': score
            }
            for dist, score in zip(self.distances, self.individual_geoscores)
        ]


class GeometricMetric(Metric):
    """地理位置度量计算器"""
    
    def __init__(self, max_distance: float = 20000.0):
        super().__init__()
        self.max_distance = max_distance
        self.geocoder = OpenCageGeocoding()
        self.add_state('distances', default_factory=list)
        self.add_state('valid_predictions', default=0)
        self.add_state('total_samples', default=0)
        self.add_state('predicted_coordinates', default_factory=list)
        self.add_state('individual_geoscores', default_factory=list)
    
    def _extract_answer_from_json(self, response: str) -> Optional[str]:
        """从JSON响应中提取answer字段"""
        try:
            # 尝试直接解析JSON
            if response.strip().startswith('{'):
                data = json.loads(response)
                if 'answer' in data:
                    return data['answer']
                elif 'FinalAnswer' in data:
                    return data['FinalAnswer']
            
            # 尝试从文本中提取JSON
            json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('answer', '')
                
            # 尝试提取Final answer
            final_answer_match = re.search(r'"FinalAnswer":\s*"([^"]*)"', response)
            if final_answer_match:
                return final_answer_match.group(1)
                
            logger.warning(f"无法从响应中提取answer: {response[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"提取answer失败: {e}")
            return None
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两个坐标点之间的距离（公里）"""
        try:
            return geodesic((lat1, lon1), (lat2, lon2)).kilometers
        except Exception as e:
            logger.error(f"计算距离失败: {e}")
            return float('inf')
    
    def update(self, predictions, labels: List[Dict]):
        """
        更新度量状态
        

        Args:
            predictions: 模型预测的JSON字符串列表
            labels: 包含真实坐标的标签列表，格式如 [{'lat_bd': x, 'lon_bd': y}, ...]
        """
        for pred, label in tqdm(zip(predictions, labels), total=len(predictions), desc="Processing predictions"):
            self.total_samples += 1
            # print('debug:判断1')

            if 'pred_lat' in pred:
                if not pred['pred_lat']:
                    self.distances.append(float('inf'))
                    self.predicted_coordinates.append(None)
                    self.individual_geoscores.append(0.0)
                    continue
                pred_lat=pred['pred_lat']
                pred_lon=pred['pred_lon']
                # 优先使用lat_bd/lon_bd, 获取不到则尝试lat/lon，最后为0
                true_lat = label.get('lat_bd', label.get('lat', 0))
                true_lon = label.get('lon_bd', label.get('lon', 0))
            
                # 保存预测坐标
                self.predicted_coordinates.append({
                    'lat': pred_lat,
                    'lon': pred_lon,
                    'address': None
                })
            
                # 计算距离
                distance = self._calculate_distance(true_lat, true_lon, pred_lat, pred_lon)
                self.distances.append(distance)
            
                # 计算单个geoscore
                if distance != float('inf'):
                    geoscore = 5000 * np.exp(-10 * (distance / self.max_distance))
                    self.valid_predictions += 1
                else:
                    geoscore = 0.0
            
                self.individual_geoscores.append(geoscore)
                continue

            
            # 提取answer字段
            answer = self._extract_answer_from_json(pred)
            if not answer:
                self.distances.append(float('inf'))
                self.predicted_coordinates.append(None)
                self.individual_geoscores.append(0.0)
                continue
            
            # 地理编码获取坐标，若失败，则逐步去掉最后一个逗号分隔的部分再试
            coords = self.geocoder.geocode(answer)
            temp_answer = answer
            if coords is None:
                # 允许最多去掉两个逗号后面的部分
                for step in range(2):
                    if ';' not in temp_answer:
                        break
                    parts = [part.strip() for part in temp_answer.split(';')]
                    if len(parts) <= 1:
                        break
                    # 若只剩下两个部分，只去一次
                    if len(parts) == 2:
                        temp_answer = '; '.join(parts[:-1])
                        coords = self.geocoder.geocode(temp_answer)
                        break
                    temp_answer = '; '.join(parts[:-1])
                    coords = self.geocoder.geocode(temp_answer)
                    if coords is not None:
                        break
            if coords is None:
                self.distances.append(float('inf'))
                self.predicted_coordinates.append(None)
                self.individual_geoscores.append(0.0)
                continue
            
            pred_lat, pred_lon = coords
            # 优先使用lat_bd/lon_bd, 获取不到则尝试lat/lon，最后为0
            true_lat = label.get('lat_bd', label.get('lat', 0))
            true_lon = label.get('lon_bd', label.get('lon', 0))
            
            # 保存预测坐标
            self.predicted_coordinates.append({
                'lat': pred_lat,
                'lon': pred_lon,
                'address': answer
            })
            
            # 计算距离
            # print('gt:',true_lat,' ',true_lon)
            # print('pd:',pred_lat,' ',pred_lon)
            distance = self._calculate_distance(true_lat, true_lon, pred_lat, pred_lon)
            self.distances.append(distance)
            
            # 计算单个geoscore
            if distance != float('inf'):
                geoscore = 5000 * np.exp(-10 * (distance / self.max_distance))
                self.valid_predictions += 1
            else:
                geoscore = 0.0
            
            self.individual_geoscores.append(geoscore)
            
            # 避免API限制
            #time.sleep(0.1)
    
    def compute(self):
        """计算最终的地理度量指标"""
        if not self.distances:
            return {}
        
        valid_distances = [d for d in self.distances if d != float('inf')]
        if not valid_distances:
            return {
                'mean_distance': 0.0,
                'median_distance': 0.0,
                'continent_accuracy': 0.0,
                'country_accuracy': 0.0,
                'region_accuracy': 0.0,
                'city_accuracy': 0.0,
                'street_accuracy': 0.0,
                'mean_geoscore': 0.0,
                'median_geoscore': 0.0,
                'valid_predictions': 0,
                'total_samples': self.total_samples
            }
        
        # 仅对有效样本（valid_distances）进行指标计算，无效样本全部排除，不纳入计算
        distances_array = np.array(valid_distances)
        num_valid = len(distances_array)

        # 基础距离指标
        if num_valid > 0:
            mean_distance = np.mean(distances_array)
            median_distance = np.median(distances_array)

            # 准确率指标
            continent_acc = np.mean(distances_array < 2500) * 100  # < 2500km
            country_acc = np.mean(distances_array < 750) * 100  # < 750km
            region_acc = np.mean(distances_array < 200) * 100  # < 200km
            city_acc = np.mean(distances_array < 25) * 100      # < 25km
            street_acc = np.mean(distances_array < 1) * 100     # < 1km
        else:
            mean_distance = 0.0
            median_distance = 0.0
            continent_acc = 0.0
            country_acc = 0.0
            region_acc = 0.0
            city_acc = 0.0
            street_acc = 0.0
        
        # GeoScore计算（只使用有效预测）
        valid_geoscores = [score for score, dist in zip(self.individual_geoscores, self.distances) if dist != float('inf')]
        if valid_geoscores:
            mean_geoscore = np.mean(valid_geoscores)
            median_geoscore = np.median(valid_geoscores)
        else:
            mean_geoscore = 0.0
            median_geoscore = 0.0
        
        return {
            'mean_distance': float(mean_distance),
            'median_distance': float(median_distance),
            'continent_accuracy': float(continent_acc),
            'country_accuracy': float(country_acc),
            'region_accuracy': float(region_acc),
            'city_accuracy': float(city_acc),
            'street_accuracy': float(street_acc),
            'mean_geoscore': float(mean_geoscore),
            'median_geoscore': float(median_geoscore),
            'valid_predictions': self.valid_predictions,
            'total_samples': self.total_samples
        }
    
    def get_predicted_coordinates(self) -> List[Optional[Dict]]:
        """
        获取所有预测的坐标信息
        
        Returns:
            坐标信息列表，每个元素是字典包含lat, lon, address，失败的预测为None
        """
        return self.predicted_coordinates.copy()
    
    def get_individual_metrics(self) -> List[Dict]:
        """
        获取每个数据的距离和geoscore
        
        Returns:
            每个数据的指标列表，包含distance和geoscore
        """
        return [
            {
                'distance': dist if dist != float('inf') else None,
                'geoscore': score
            }
            for dist, score in zip(self.distances, self.individual_geoscores)
        ]


def compute_geometric_metrics(prediction) -> Dict[str, float]:
    """
    计算地理位置度量的主函数
    
    Args:
        prediction: 包含(predictions, labels)的tuple
        
    Returns:
        包含所有地理度量指标的字典
    """
    preds, labels = prediction[0], prediction[1]
    
    # 将tensor转换为字符串列表
    new_preds = []
    for i in range(preds.shape[0]):
        pred_str = Serializer.from_tensor(preds[i])
        new_preds.append(pred_str)
    
    # 将tensor转换为标签字典列表
    new_labels = []
    for i in range(labels.shape[0]):
        label_str = Serializer.from_tensor(labels[i])
        try:
            # 解析标签格式: 既可能有bd字段，也可能只有lat/lon，最终结构统一为{'lat':..., 'lon':...}（不带bd）
            label_dict = json.loads(label_str)
            if 'content' in label_dict:
                # content内容可以是json字符串或直接是dict
                content_val = label_dict['content']
                if isinstance(content_val, str):
                    try:
                        content_dict = json.loads(content_val)
                    except Exception:
                        content_dict = {}
                elif isinstance(content_val, dict):
                    content_dict = content_val
                else:
                    content_dict = {}

                # 优先lat_bd/lon_bd, 其次lat/lon
                lat = content_dict.get('lat_bd', content_dict.get('lat', 0))
                lon = content_dict.get('lon_bd', content_dict.get('lon', 0))
                new_labels.append({'lat': lat, 'lon': lon})
            else:
                # 外部格式，优先lat_bd/lon_bd, 其次lat/lon
                lat = label_dict.get('lat_bd', label_dict.get('lat', 0))
                lon = label_dict.get('lon_bd', label_dict.get('lon', 0))
                new_labels.append({'lat': lat, 'lon': lon})

        except Exception as e:
            # 如果解析失败，跳过这个样本
            logger.warning(f"无法解析标签: {label_str[:100]}..., 错误: {e}")
            new_labels.append({'lat': 0, 'lon': 0})
    
    # 创建度量计算器并计算
    metric = GeometricMetric()
    metric.update(new_preds, new_labels)
    return metric.compute()


def compute_rouge_bleu(preds: List[str], labels: List[str]):
    import jieba
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge.rouge import Rouge
    score_dict = {key: MeanMetric() for key in ['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']}

    for pred, label in zip(preds, labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        if not hypothesis or not reference:
            continue
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))[0]
        for k, v in scores.items():
            score_dict[k].update(v['f'])
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict['bleu-4'].update(bleu_score)

    return {k: round(v.compute()['value'] * 100, 6) for k, v in score_dict.items()}


def compute_nlg_metrics(prediction) -> Dict[str, float]:
    preds, labels = prediction[0], prediction[1]
    new_preds, new_labels = [], []
    for i in range(preds.shape[0]):
        new_preds.append(Serializer.from_tensor(preds[i]))
        new_labels.append(Serializer.from_tensor(labels[i]))
    return compute_rouge_bleu(new_preds, new_labels)


def compute_acc(preds,
                labels,
                *,
                acc_strategy: Literal['token', 'seq'] = 'token',
                is_encoder_decoder: bool = False) -> Dict[str, List[float]]:

    if isinstance(preds, torch.Tensor):
        if torch.is_floating_point(labels):
            return {}
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    if preds.ndim >= 2 and not is_encoder_decoder:
        labels = labels[..., 1:]
        preds = preds[..., :-1]
    if np.issubdtype(labels.dtype, np.floating) or preds.shape != labels.shape:
        return {}

    masks = labels != -100
    if acc_strategy == 'token' or preds.ndim == 1:
        acc_list = (preds[masks] == labels[masks]).tolist()
    else:
        acc_list = []
        for i, m in enumerate(masks):
            acc_list.append(np.all(preds[i, m] == labels[i, m]))
    return {f'{acc_strategy}_acc' if preds.ndim >= 2 else 'acc': acc_list}


def compute_acc_metrics(eval_prediction: EvalPrediction,
                        *,
                        acc_strategy: Literal['token', 'seq'] = 'token',
                        is_encoder_decoder: bool = False) -> Dict[str, float]:

    metric = compute_acc(
        eval_prediction.predictions,
        eval_prediction.label_ids,
        acc_strategy=acc_strategy,
        is_encoder_decoder=is_encoder_decoder)
    if len(metric) == 0:
        return {}
    return {k: sum(v) / len(v) for k, v in metric.items()}


def preprocess_logits_for_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = logits.argmax(dim=-1)
    return preds


# Add your own metric calculation method here, use --metric xxx to train
METRIC_MAPPING = {
    'acc': (compute_acc_metrics, preprocess_logits_for_acc),
    'nlg': (compute_nlg_metrics, None),
    # 'geometric': (compute_geometric_metrics, None),
}


def get_metric(metric: str):
    return METRIC_MAPPING[metric]
