### Input
{
    number_of_images : Number of images to generate
    input_gender : One of - "Men", "Women"
    input_category : One of - "SHIRTS", "SWEATERS", "JEANS", "PANTS",  "TOPS", "SUITS & BLAZERS", "SHORTS", "JACKETS & COATS", "SKIRTS", "JUMPSUITS", "SWIMWEAR", "DRESSES"
    input_pose : One of - "id_gridfs_1", "id_gridfs_2", "id_gridfs_3", "id_gridfs_4"
}



### Example
```
curl -X POST -H "Content-Type: application/json" -d '{"number_of_images":10,"input_gender":"Men","input_category":"SHIRTS", "input_pose":"id_gridfs_1"}' http://localhost:8080/predictions/dcgan_fashiongen -o test_img1.jpg

curl -X POST -H "Content-Type: application/json" -d '{"number_of_images":20,"input_gender":"Women","input_category":"DRESSES", "input_pose":"id_gridfs_3"}' http://localhost:8080/predictions/dcgan_fashiongen -o test_img2.jpg

```
