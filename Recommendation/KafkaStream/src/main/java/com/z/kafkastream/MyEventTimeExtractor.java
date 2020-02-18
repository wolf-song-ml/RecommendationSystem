package com.z.kafkastream;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.streams.processor.TimestampExtractor;

/**
 * kafka 0.10以下生产消息没有时间戳，flume使用kafka版本较低
 */
public class MyEventTimeExtractor implements TimestampExtractor{

    @Override
    public long extract(ConsumerRecord<Object, Object> record, long previousTimestamp) {
        return System.currentTimeMillis();
    }
}
