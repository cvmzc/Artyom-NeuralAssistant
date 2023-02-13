package main

import (
	"fmt"
	"net/http"
	"log"
)

func sayhello(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Привет!")
}

func main() {
	http.HandleFunc("/", sayhello) // Устанавливаем роутер
	err := http.ListenAndServe(":80", nil) // устанавливаем порт веб-сервера

	// Если хотите использовать https, то вместо ListenAndServe используйте ListenAndServeTLS
	// err := http.ListenAndServeTLS(":8080", "cert.pem", "key.pem", nil)
	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}