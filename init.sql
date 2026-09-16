-- 强制以 utf8mb4 解析本文件,避免 MySQL 客户端默认字符集不是 utf8mb4 时
-- (如部分容器环境默认 latin1/cp1252),中文被双重编码成乱码写入
SET NAMES utf8mb4;

-- 创建案例表
CREATE TABLE IF NOT EXISTS legal_cases (
    id INT AUTO_INCREMENT PRIMARY KEY,
    case_name VARCHAR(255) NOT NULL,        -- 案件名称
    case_type VARCHAR(100),                 -- 案件类型（合同纠纷、刑事等）
    court VARCHAR(255),                     -- 审理法院
    judgment_date DATE,                     -- 判决日期
    summary TEXT,                           -- 案件摘要
    keywords VARCHAR(500),                  -- 关键词，便于检索
    full_text TEXT                          -- 全文（可选）
);

-- 插入示例数据
INSERT INTO legal_cases (case_name, case_type, court, judgment_date, summary, keywords) VALUES
('张三诉李四合同纠纷案', '合同纠纷', '北京市朝阳区人民法院', '2023-05-10', '双方签订买卖合同，被告未按期付款，法院判决被告支付货款及违约金。', '合同,违约,货款,违约金'),
('王五盗窃案', '刑事', '上海市浦东新区人民法院', '2023-08-15', '被告人多次盗窃他人财物，数额较大，判处有期徒刑一年。', '盗窃,刑事,有期徒刑');