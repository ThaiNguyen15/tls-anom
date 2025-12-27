FROM zeek/zeek:7.0.8

# Install JA3 package
RUN zkg refresh && yes | zkg install zeek/salesforce/ja3 && zkg load ja3

RUN echo "@load ja3" >> /usr/local/zeek/share/zeek/site/local.zeek

RUN cp . .
CMD ["./entrypoint.sh"]
