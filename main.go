package main

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"net/http/cookiejar"
	"os"
	"strconv"
	"time"
)

const (
	recaptchaUrl = "https://portal.aut.ac.ir/aportal/PassImageServlet"
	portalUrl    = "https://portal.aut.ac.ir/aportal/loginRedirect.jsp"
	loginUrl     = "https://portal.aut.ac.ir/aportal/login.jsp"
)

const (
	SAME_IMAGE      = 20
	DIFFERENT_IMAGE = 16
)

func main() {
	login("2312", "sdfsdfds")
	//creatingData()
}

func login(username string, password string) {
	jar, err := cookiejar.New(nil)
	if err != nil {
		fmt.Printf("error making cookiejar: %s\n", err)
		os.Exit(1)
	}

	client := http.Client{
		Timeout: 30 * time.Second,
		Jar:     jar,
	}

	err = settingCaptchaCookie(&client)
	if err != nil {
		fmt.Printf("SettingCapthcaCookie error: %s\n", err)
		os.Exit(1)
	}

	captcha, err := solvingCaptcha(&client, "captcha")
	if err != nil {
		fmt.Printf("trying to solve captcha error: %s\n", err)
		os.Exit(1)
	}

	err = sendLoginRequest(&client, captcha, username, password)
	if err != nil {
		fmt.Printf("trying to login: %s\n", err)
		os.Exit(1)
	}
}

func sendLoginRequest(client *http.Client, captcha string, username string, password string) error {
	// Create a buffer to write the request body
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	err := writer.WriteField("login", "ورود به پورتال")
	if err != nil {
		return err
	}
	err = writer.WriteField("username", username)
	if err != nil {
		return err
	}
	err = writer.WriteField("password", password)
	if err != nil {
		return err
	}
	err = writer.WriteField("passline", captcha)
	if err != nil {
		return err
	}
	err = writer.Close()
	if err != nil {
		return err
	}

	redirectReq, err := http.NewRequest(http.MethodPost, loginUrl, &requestBody)
	redirectReq.Header.Set("Host", "portal.aut.ac.ir")
	redirectReq.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
	redirectReq.Header.Set("Connection", "keep-alive")
	redirectReq.Header.Set("Content-Type", writer.FormDataContentType())
	redirectReq.Header.Set("Content-Length", strconv.FormatInt(int64(requestBody.Len()), 10))

	res, err := client.Do(redirectReq)
	if err != nil {
		return errors.New(fmt.Sprintf("error making http request: %s\n", err))
	}
	if res.Header.Get("Vary") == "" {
		return errors.New("error not acceptable cookie")
	}
	resBody, err := ioutil.ReadAll(res.Body)
	if err != nil {
		fmt.Printf("server: could not read request body")
	}
	notEnough := []byte("حروف تصویر کافی نمیباشد...")
	wrongCaptcha := []byte("حروف تصویر صحیح نمیباشد...")
	if bytes.Contains(resBody, notEnough) || bytes.Contains(resBody, wrongCaptcha) {
		return errors.New("wrong captcha")
	}
	wrongUserData := []byte("اطلاعات وارد شده نامعتبر است!")
	if bytes.Contains(resBody, wrongUserData) {
		return errors.New("wrong user data")
	}
	siteIsClosed := []byte("فقط در زمان ثبت نام کارشناسی")
	if bytes.Contains(resBody, siteIsClosed) {
		return errors.New("site is closed")
	}
	return nil
}

func solvingCaptcha(client *http.Client, imageName string) (string, error) {
	text := ""
	var counter int
	for counter = 1; text == "" && counter < 20; counter++ {

		captchaBody, err := sendingCaptchaRequest(client)
		if err != nil {
			return "", errors.New(fmt.Sprintf("GettingCaptchaResponse error: %s\n", err))
		}

		err = storingImage(captchaBody, "captcha")
		if err != nil {
			return "", errors.New(fmt.Sprintf("StoringImage error: %s\n", err))
		}
		// TODO fill it with getting form ai
		text = "something"
	}
	if counter >= 20 {
		return "", errors.New("too many attempt to solve this captcha")
	}
	return text, nil
}

func creatingData() {
	for i := 0; i < DIFFERENT_IMAGE; i++ {
		jar, err := cookiejar.New(nil)
		if err != nil {
			fmt.Printf("error making cookiejar: %s\n", err)
			os.Exit(1)
		}

		client := http.Client{
			Timeout: 30 * time.Second,
			Jar:     jar,
		}
		err = settingCaptchaCookie(&client)
		if err != nil {
			fmt.Printf("SettingCapthcaCookie error: %s\n", err)
			os.Exit(1)
		}

		for j := 0; j < SAME_IMAGE; j++ {
			captchaBody, err := sendingCaptchaRequest(&client)
			if err != nil {
				fmt.Printf("GettingCaptchaResponse error: %s\n", err)
				os.Exit(1)
			}

			err = storingImage(captchaBody, fmt.Sprintf("./data/%d_%d", i, j))
			if err != nil {
				fmt.Printf("StoringImage error: %s\n", err)
				os.Exit(1)
			}
		}
	}
}

func settingCaptchaCookie(client *http.Client) error {

	// Create a buffer to write the request body
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	err := writer.WriteField("login", "ورود از درگاه قبلي پورتال")
	if err != nil {
		return err
	}

	err = writer.Close()
	if err != nil {
		return err
	}

	redirectReq, err := http.NewRequest(http.MethodGet, portalUrl, &requestBody)
	redirectReq.Header.Set("Host", "portal.aut.ac.ir")
	redirectReq.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
	redirectReq.Header.Set("Connection", "keep-alive")
	redirectReq.Header.Set("Content-Type", writer.FormDataContentType())
	redirectReq.Header.Set("Content-Length", strconv.FormatInt(int64(requestBody.Len()), 10))

	res, err := client.Do(redirectReq)
	if err != nil {
		return errors.New(fmt.Sprintf("error making http request: %s\n", err))
	}
	if res.Header.Get("Vary") == "" {
		return errors.New("error not acceptable cookie")
	}
	return nil
}

func sendingCaptchaRequest(client *http.Client) ([]byte, error) {

	recaptchaImageReg, err := http.NewRequest(http.MethodGet, recaptchaUrl, nil)
	recaptchaImageReg.Header.Set("Host", "portal.aut.ac.ir")
	recaptchaImageReg.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
	recaptchaImageReg.Header.Set("Connection", "keep-alive")

	res, err := client.Do(recaptchaImageReg)
	if err != nil {
		return nil, errors.New(fmt.Sprintf("error making http request: %s\n", err))
	}

	resBody, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return nil, errors.New(fmt.Sprintf("login(getting csrf): could not read response body: %s\n", err))
	}
	return resBody, nil
}

func storingImage(image []byte, name string) error {
	fullName := name + ".jpg"
	err := os.Remove(fullName)
	if err != nil && err.Error() != "remove "+fullName+": The system cannot find the file specified." {
		return err
	}
	err = ioutil.WriteFile(fullName, image, 0)
	if err != nil {
		return err
	}
	return nil
}
