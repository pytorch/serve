# SSD Example Outputs

### Dog Beach

![dog beach](https://farm9.staticflickr.com/8184/8081332083_3a5c242b8b_z_d.jpg)
```bash
curl -o dogbeach.jpg https://farm9.staticflickr.com/8184/8081332083_3a5c242b8b_z_d.jpg
curl -X POST http://127.0.0.1:8080/ssd/predict -F "data=@dogbeach.jpg"
{
  "prediction": [
    [
      "person",
      203,
      213,
      248,
      347
    ],
    [
      "dog",
      334,
      175,
      403,
      235
    ],
    [
      "person",
      109,
      211,
      144,
      291
    ],
    [
      "person",
      529,
      31,
      562,
      103
    ],
    [
      "person",
      155,
      12,
      189,
      98
    ],
    [
      "horse",
      465,
      3,
      527,
      40
    ],
    [
      "person",
      51,
      372,
      96,
      427
    ],
    [
      "dog",
      80,
      56,
      131,
      96
    ],
    [
      "person",
      70,
      89,
      96,
      155
    ],
    [
      "cow",
      292,
      188,
      344,
      231
    ],
    [
      "dog",
      294,
      186,
      349,
      231
    ]
  ]
}
```

### 3 Dogs on Beach
![3 dogs on beach](https://farm9.staticflickr.com/8051/8081326814_64756479c6_z_d.jpg)
```bash
curl -o 3dogs.jpg https://farm9.staticflickr.com/8051/8081326814_64756479c6_z_d.jpg
curl -X POST http://127.0.0.1:8080/ssd/predict -F "data=@3dogs.jpg"
{
  "prediction": [
    [
      "dog",
      399,
      128,
      570,
      290
    ],
    [
      "dog",
      278,
      196,
      417,
      286
    ],
    [
      "cow",
      205,
      116,
      297,
      272
    ]
  ]
}
```
### Sailboat
![sailboat](https://farm9.staticflickr.com/8316/7990362092_84a688a089_z_d.jpg)
```bash
curl -o sailboat.jpg https://farm9.staticflickr.com/8316/7990362092_84a688a089_z_d.jpg
curl -X POST http://127.0.0.1:8080/ssd/predict -F "data=@sailboat.jpg"
{
  "prediction": [
    [
      "boat",
      160,
      87,
      249,
      318
    ]
  ]
}
```
