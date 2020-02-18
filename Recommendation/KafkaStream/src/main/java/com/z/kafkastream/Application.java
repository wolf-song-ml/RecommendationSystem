package com.z.kafkastream;

import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.processor.TopologyBuilder;

import java.util.Properties;

/**
 * kafka stream实时流
 */
public class Application {
    public static void main(String[] args) {
        String brokers = "node1:9092,node2:9092,node3:9092";
//        String zookeepers = "node1:2181,node2:2181,node3:2181";

        // topic
        String from = "log";
        String to = "recommender";

        // kafka消费者配置
        Properties settings = new Properties();
        settings.put(StreamsConfig.APPLICATION_ID_CONFIG, "logFilter");
        settings.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, brokers);
        // flume端的kafka是0.8.11版本，兼容发送没有发送时间戳
        settings.put(StreamsConfig.TIMESTAMP_EXTRACTOR_CLASS_CONFIG, MyEventTimeExtractor.class.getName());
//        settings.put(StreamsConfig.ZOOKEEPER_CONNECT_CONFIG, zookeepers);

     // 创建 kafka stream 配置对象
        StreamsConfig config = new StreamsConfig(settings);

     // 创建一个拓扑建构器
        TopologyBuilder builder = new TopologyBuilder();

     // 定义流处理的拓扑结构
        builder.addSource("SOURCE", from)
                .addProcessor("PROCESSOR", ()-> new LogProcessor(), "SOURCE")
                .addSink("SINK", to, "PROCESSOR");

        KafkaStreams streams = new KafkaStreams( builder, config );

        streams.start();

        System.out.println("Kafka stream started!>>>>>>>>>>>");

    }
}
